# calibrate_triadic.py
import json
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def generate_calibration_hook(layer_idx, telemetry_ledger):
    """Captures undisturbed baseline metrics on the forward pass."""
    def hook(module, input, output):
        # Extract tensor if output is a tuple (standard for layer blocks)
        x = output[0] if isinstance(output, tuple) else output
        x_fp32 = x.to(torch.float32).detach()
        
        variance = torch.std(x_fp32).item()
        mean = torch.mean(x_fp32).item()
        density = (x_fp32 > 0).float().mean().item() * 100.0
        
        probs = F.softmax(x_fp32, dim=-1)
        entropy = -torch.sum(probs * torch.log2(probs + 1e-9), dim=-1).mean().item()
        
        telemetry_ledger[layer_idx]['variance'].append(variance)
        telemetry_ledger[layer_idx]['mean'].append(mean)
        telemetry_ledger[layer_idx]['density'].append(density)
        telemetry_ledger[layer_idx]['entropy'].append(entropy)
        
    return hook

def run_calibration_loop():
    print("Initializing Unmodified Llama-3.2-1B Calibration...")
    model_id = "meta-llama/Llama-3.2-1B"
    device = "cuda" if torch.cuda.is_available() else "mps"
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize ledger for 16 layers
    telemetry_ledger = {i: {'variance': [], 'mean': [], 'density': [], 'entropy': []} for i in range(16)}
    hooks = []

    # Map telemetry probes to input_layernorm
    for i in range(16):
        target = model.model.layers[i].input_layernorm
        h = target.register_forward_hook(generate_calibration_hook(i, telemetry_ledger))
        hooks.append(h)

    # Ingest a standardized subset of GSM8K for logical reasoning baselines
    print("Streaming GSM8K Calibration Batch...")
    dataset = load_dataset("openai/gsm8k", "main", split="train[:250]")
    
    for item in dataset:
        inputs = tokenizer(item["question"], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            model(**inputs)

    # Strip probes
    for h in hooks:
        h.remove()

    # Calculate empirical baselines and serialize to JSON
    print("Compiling Layer Variance Configurations...")
    layer_config = {}
    for layer_idx, metrics in telemetry_ledger.items():
        layer_config[f"layer_{layer_idx}"] = {
            "sigma_baseline": float(np.mean(metrics['variance'])),
            "mu_baseline": float(np.mean(metrics['mean'])),
            "natural_density": float(np.mean(metrics['density'])),
            "spatial_entropy": float(np.mean(metrics['entropy'])),
            # Derive specific constraint thresholds dynamically from empirical variance
            "tau_perimeter": float(np.mean(metrics['variance']) * 0.15) 
        }

    with open("triadic_layer_config.json", "w") as f:
        json.dump(layer_config, f, indent=4)
    print("Calibration Ledger Exported: triadic_layer_config.json")

if __name__ == "__main__":
    run_calibration_loop()
