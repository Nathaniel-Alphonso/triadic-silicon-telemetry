import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

def triadic_clamp_hook(module, args, output):
    tau = 0.05
    x = output[0] if isinstance(output, tuple) else output
    orig_dtype = x.dtype
    
    x_clamped = x.to(torch.float32)
    omega_mask = (x_clamped > tau).to(dtype=x_clamped.dtype)
    x_clamped = x_clamped * omega_mask
    
    if isinstance(output, tuple):
        return (x_clamped.to(orig_dtype),) + output[1:]
    return x_clamped.to(orig_dtype)

def run_latency_benchmark():
    model_id = "meta-llama/Llama-3.2-1B"
    print("Loading Llama-3.2-1B for streaming...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to("mps")
    
    prompt = "Explain the fundamental principles of quantum entanglement in a clear, concise paragraph:"
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")
    
    def measure_pass(hook_enabled=False):
        handle = None
        if hook_enabled:
            target_layer = model.model.layers[12].self_attn.o_proj
            handle = target_layer.register_forward_hook(triadic_clamp_hook)
            
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        
        # Flush the MPS queue before starting the clock to prevent queued kernel artifacts
        torch.mps.synchronize()
        
        t_start = time.perf_counter()
        thread.start()
        
        token_times = []
        for _ in streamer:
            token_times.append(time.perf_counter())
            
        thread.join()
        if handle is not None:
            handle.remove()
            
        ttft_ms = (token_times[0] - t_start) * 1000
        itls = [(token_times[i] - token_times[i-1]) * 1000 for i in range(1, len(token_times))]
        mean_itl_ms = sum(itls) / len(itls) if itls else 0.0
        
        return ttft_ms, mean_itl_ms

    print("\nWarming up MPS backend...")
    measure_pass(hook_enabled=False)
    
    print("\nExecuting Unhooked Baseline...")
    base_ttft, base_itl = measure_pass(hook_enabled=False)
    
    print("Executing Triadic Hooked Pass...")
    hook_ttft, hook_itl = measure_pass(hook_enabled=True)
    
    print("\n--- FINAL THROUGHPUT DELTAS ---")
    print(f"Baseline TTFT: {base_ttft:.2f} ms")
    print(f"Hooked TTFT:   {hook_ttft:.2f} ms")
    print(f"TTFT Delta:    {hook_ttft - base_ttft:+.2f} ms")
    print("-------------------------------")
    print(f"Baseline ITL:  {base_itl:.2f} ms/token")
    print(f"Hooked ITL:    {hook_itl:.2f} ms/token")
    print(f"ITL Delta:     {hook_itl - base_itl:+.2f} ms/token\n")

if __name__ == "__main__":
    run_latency_benchmark()
