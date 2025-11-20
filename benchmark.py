import json
import time
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import threading
import gc
import traceback

# Load configurations
with open('models_config.json', 'r') as f:
    models_config = json.load(f)

with open('devices_config.json', 'r') as f:
    devices_config = json.load(f)

# Import utility functions
from pruning_utils import apply_structured_pruning, apply_magnitude_pruning
from lora_utils import apply_lora

class MemoryMonitor:
    def __init__(self):
        self.peak_memory = 0
        self.stop_monitoring = False
    
    def monitor(self):
        while not self.stop_monitoring:
            current_memory = psutil.Process().memory_info().rss / (1024**3)
            self.peak_memory = max(self.peak_memory, current_memory)
            time.sleep(0.1)
    
    def start(self):
        self.monitor_thread = threading.Thread(target=self.monitor)
        self.monitor_thread.start()
    
    def stop(self):
        self.stop_monitoring = True
        self.monitor_thread.join()
        return self.peak_memory

def load_model_quantized(model_name, quantization):
    """Load model with specific quantization"""
    print(f"Loading {model_name} with {quantization}...")
    
    try:
        if quantization == "FP16":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="cpu"
            )
        elif quantization == "INT8":
            # Use BitsAndBytesConfig for INT8
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="cpu"
            )
        elif quantization == "INT4":
            # Use BitsAndBytesConfig for INT4
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="cpu"
            )
        else:
            raise ValueError(f"Unsupported quantization: {quantization}")
        
        return model
    
    except Exception as e:
        print(f"Error loading model {model_name} with {quantization}: {e}")
        raise

def benchmark_model(model_config, device_config, quantization):
    """Run full benchmark for a model/device/quantization combination"""
    results = {
        "model": model_config["name"],
        "parameters": model_config["parameters"],
        "device": device_config["name"],
        "quantization": quantization,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:
        # Check if quantization is supported
        if quantization not in model_config["supported_quant"]:
            results["status"] = "unsupported_quant"
            return results
        
        # Check memory constraint
        estimated_memory = model_config["parameters"] * 2  # Rough estimate for FP16
        if quantization == "INT8":
            estimated_memory /= 2
        elif quantization == "INT4":
            estimated_memory /= 4
        
        if estimated_memory > device_config["ram_gb"] * 0.8:
            results["status"] = "memory_exceeded"
            return results
        
        # Load tokenizer
        print(f"  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_config["name"])
        
        # Load model
        model = load_model_quantized(model_config["name"], quantization)
        
        # Create pipeline
        print(f"  Creating pipeline...")
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="cpu"
        )
        
        # Test prompts
        test_prompts = [
            "What is the capital of France?",
            "Explain photosynthesis in simple terms.",
            "Write a Python function to reverse a string."
        ]
        
        # Benchmark each prompt
        all_latencies = []
        total_memory_gb = 0
        
        for i, prompt in enumerate(test_prompts):
            print(f"  Testing prompt {i+1}/3: {prompt[:50]}...")
            
            # Start memory monitoring
            monitor = MemoryMonitor()
            monitor.start()
            
            # Measure latency
            start_time = time.time()
            output = pipe(prompt, max_new_tokens=50, do_sample=False)
            end_time = time.time()
            
            # Stop memory monitoring
            peak_memory = monitor.stop()
            total_memory_gb = max(total_memory_gb, peak_memory)
            
            # Calculate latency
            latency_ms = (end_time - start_time) * 1000
            all_latencies.append(latency_ms)
            
            # Cleanup
            del output
            gc.collect()
        
        # Average metrics
        avg_latency = sum(all_latencies) / len(all_latencies)
        
        results["status"] = "success"
        results["avg_latency_ms"] = round(avg_latency, 2)
        results["peak_memory_gb"] = round(total_memory_gb, 2)
        results["tokens_per_second"] = round(50 / (avg_latency / 1000), 2)
        
        # Test pruning (if supported)
        if model_config["supports_pruning"] and quantization in ["FP16", "INT8"]:
            print("  Testing: Pruning (30%)")
            try:
                # Recreate model for pruning test
                model_for_pruning = load_model_quantized(model_config["name"], quantization)
                pruned_model = apply_structured_pruning(model_for_pruning, 0.3)
                
                # Create pipeline for pruned model
                pruned_pipe = pipeline(
                    "text-generation",
                    model=pruned_model,
                    tokenizer=tokenizer,
                    device_map="cpu"
                )
                
                # Measure pruned model latency
                monitor = MemoryMonitor()
                monitor.start()
                start_time = time.time()
                output = pruned_pipe(test_prompts[0], max_new_tokens=50, do_sample=False)
                end_time = time.time()
                peak_memory_pruned = monitor.stop()
                
                results["pruned_latency_ms"] = round((end_time - start_time) * 1000, 2)
                results["pruned_memory_gb"] = round(peak_memory_pruned, 2)
                results["pruned_status"] = "success"
                
                del model_for_pruning, pruned_model, pruned_pipe
                gc.collect()
            except Exception as e:
                print(f"    Pruning failed: {e}")
                results["pruned_status"] = "failed"
                results["pruned_error"] = str(e)
        
        # Test LoRA (if supported)
        if model_config["supports_lora"] and quantization in ["FP16", "INT8"]:
            print("  Testing: LoRA")
            try:
                # Recreate model for LoRA test
                model_for_lora = load_model_quantized(model_config["name"], quantization)
                lora_model = apply_lora(model_for_lora)
                
                # Create pipeline for LoRA model
                lora_pipe = pipeline(
                    "text-generation",
                    model=lora_model,
                    tokenizer=tokenizer,
                    device_map="cpu"
                )
                
                # Measure LoRA latency
                monitor = MemoryMonitor()
                monitor.start()
                start_time = time.time()
                output = lora_pipe(test_prompts[0], max_new_tokens=50, do_sample=False)
                end_time = time.time()
                peak_memory_lora = monitor.stop()
                
                results["lora_latency_ms"] = round((end_time - start_time) * 1000, 2)
                results["lora_memory_gb"] = round(peak_memory_lora, 2)
                results["lora_status"] = "success"
                
                del model_for_lora, lora_model, lora_pipe
                gc.collect()
            except Exception as e:
                print(f"    LoRA failed: {e}")
                results["lora_status"] = "failed"
                results["lora_error"] = str(e)
        
        # Cleanup base model
        del model, pipe, tokenizer
        gc.collect()
        
    except Exception as e:
        print(f"ERROR in benchmark: {e}")
        results["status"] = "error"
        results["error_message"] = str(e)
        results["traceback"] = traceback.format_exc()
    
    return results

def main():
    """Main benchmark loop"""
    print("="*60)
    print("EdgeLLM Benchmark Suite")
    print("="*60)
    
    all_results = []
    
    # Loop through each device
    for device in devices_config["devices"]:
        print(f"\n{'='*60}")
        print(f"Benchmarking on {device['name']}")
        print(f"RAM: {device['ram_gb']}GB, CPUs: {device['cpus']}")
        print(f"{'='*60}")
        
        # Loop through each model
        for model in models_config["models"]:
            print(f"\n--- Testing Model: {model['name']} ---")
            
            # Loop through each quantization
            for quant in model["supported_quant"]:
                print(f"  Quantization: {quant}")
                
                result = benchmark_model(model, device, quant)
                all_results.append(result)
                
                # Print result
                if result["status"] == "success":
                    print(f"    ✓ Latency: {result['avg_latency_ms']}ms")
                    print(f"    ✓ Memory: {result['peak_memory_gb']}GB")
                    print(f"    ✓ Speed: {result['tokens_per_second']} tokens/sec")
                    
                    if "pruned_status" in result and result["pruned_status"] == "success":
                        print(f"    ✓ Pruned Latency: {result['pruned_latency_ms']}ms")
                    
                    if "lora_status" in result and result["lora_status"] == "success":
                        print(f"    ✓ LoRA Latency: {result['lora_latency_ms']}ms")
                else:
                    print(f"    ✗ Status: {result['status']}")
                    if "error_message" in result:
                        print(f"    Error: {result['error_message']}")
                
                # Small delay between runs
                time.sleep(2)
        
        print(f"\nDevice {device['name']} complete!\n")
    
    # Save results
    output_file = 'benchmark_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"{'='*60}")
    print("Benchmarking Complete!")
    print(f"Results saved to {output_file}")
    print(f"Total configurations tested: {len(all_results)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
