import json
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def compute_l1_entropy(x_fp32: torch.Tensor) -> float:
    """Calculates spatial entropy using L1-normalized absolute magnitudes."""
    abs_x = torch.abs(x_fp32)
    p_x = abs_x / (torch.sum(abs_x, dim=-1, keepdim=True) + 1e-9)
    entropy = -torch.sum(p_x * torch.log2(p_x + 1e-9), dim=-1)
    return entropy.mean().item()

def generate_calibration_hook(key_name, telemetry_ledger, inspect_input=False):
    """Captures baseline metrics on the forward pass."""
    def hook(module, input, output):
        if inspect_input and input:
            x = input[0]
        else:
            x = output[0] if isinstance(output, tuple) else output
            
        x_fp32 = x.to(torch.float32).detach()
        
        # Computed feature-wise to prevent sequence-length dimensional distortion
        std_dev = torch.std(x_fp32, dim=-1).mean().item()
        mean = torch.mean(x_fp32, dim=-1).mean().item()
        density = (x_fp32 > 0).float().mean(dim=-1).mean().item() * 100.0
        entropy = compute_l1_entropy(x_fp32)
        
        telemetry_ledger[key_name]['std_dev'].append(std_dev)
        telemetry_ledger[key_name]['mean'].append(mean)
        telemetry_ledger[key_name]['density'].append(density)
        telemetry_ledger[key_name]['entropy'].append(entropy)
        
    return hook

def run_calibration_loop():
    print("Initializing Unmodified Llama-3.2-1B-Instruct Calibration...")
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "mps"
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    keys = ["embed_tokens"] + [f"layer_{i}" for i in range(16)] + ["lm_head"]
    telemetry_ledger = {k: {'std_dev': [], 'mean': [], 'density': [], 'entropy': []} for k in keys}
    hooks = []

    h_embed = model.model.embed_tokens.register_forward_hook(
        generate_calibration_hook("embed_tokens", telemetry_ledger, inspect_input=False)
    )
    hooks.append(h_embed)

    for i in range(16):
        target = model.model.layers[i].input_layernorm
        h = target.register_forward_hook(generate_calibration_hook(f"layer_{i}", telemetry_ledger, inspect_input=True))
        hooks.append(h)

    h_lm_head = model.lm_head.register_forward_hook(
        generate_calibration_hook("lm_head", telemetry_ledger, inspect_input=False)
    )
    hooks.append(h_lm_head)

    print("Streaming GSM8K Calibration Batch with Chat Formatting...")
    dataset = load_dataset("openai/gsm8k", "main", split="train[:250]")
    
    for item in dataset:
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["question"]}],
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            model(**inputs)

    for h in hooks:
        h.remove()

    print("Compiling Baseline Telemetry Configuration Ledger...")
    layer_config = {}
    for key_name, metrics in telemetry_ledger.items():
        layer_config[key_name] = {
            "sigma_baseline": float(np.mean(metrics['std_dev'])),
            "mu_baseline": float(np.mean(metrics['mean'])),
            "natural_density": float(np.mean(metrics['density'])),
            "spatial_entropy": float(np.mean(metrics['entropy'])),
            "tau_perimeter": float(np.mean(metrics['std_dev']) * 0.15) 
        }

    with open("triadic_layer_config.json", "w") as f:
        json.dump(layer_config, f, indent=4)
    print("Calibration Ledger Exported: triadic_layer_config.json")

if __name__ == "__main__":
    run_calibration_loop()
