import time
import psutil
from llama_cpp import Llama
import os

# --- CONFIGURE YOUR TEST HERE ---

# 1. Point this to the model you mounted inside the container
# We will mount our models to /models
# !! IMPORTANT: Update this to a model you have downloaded !!
MODEL_PATH = "/models/phi-3-mini-4k-instruct-q4_K_M.gguf"

# 2. Add queries specific to YOUR use case (search & NLP)
# This is the most important part. Use realistic data.
TEST_QUERIES = [
    "Search for recent news on AI in healthcare and provide three bullet points.",
    "What is the capital of France?",
    "Summarize the main points of the article about quantum computing.",
    "Explain the concept of 'state space models' in simple terms.",
    "Write a polite email asking for a follow-up meeting."
]

# 3. Model parameters - CONFIGURABLE
LLM_PARAMS = {
    "n_ctx": 2048,      # Context size
    "n_threads": 4,     # Manually set thread count (match your simulated CPUs)
    "n_gpu_layers": 0   # Explicitly set to 0 to ensure CPU-only
}
# --- END OF CONFIGURATION ---


def get_process_memory():
    """Get the current memory usage of this Python process."""
    process = psutil.Process(os.getpid())
    # Return in Megabytes (MB)
    return process.memory_info().rss / (1024 * 1024)


def run_benchmark():
    """
    Runs the full benchmark suite for the specified model.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Did you mount the models volume correctly in Docker?")
        print("Please update the MODEL_PATH variable in benchmark.py")
        return

    print(f"--- Starting Benchmark for: {os.path.basename(MODEL_PATH)} ---")
    print(f"--- CPU-Only (n_gpu_layers=0), Threads={LLM_PARAMS.get('n_threads')} ---")

    # --- 1. Model Load Test ---
    print("\n[Phase 1: Model Loading]")
    start_load_time = time.time()
    
    # Get memory before loading
    mem_before_load = get_process_memory()
    
    llm = Llama(model_path=MODEL_PATH, **LLM_PARAMS)
    
    # Get memory after loading
    mem_after_load = get_process_memory()
    
    load_time = time.time() - start_load_time
    model_ram_usage = mem_after_load - mem_before_load

    print(f"Model Load Time: {load_time:.2f} seconds")
    print(f"Model RAM Footprint: {model_ram_usage:.2f} MB")

    # --- 2. Inference Test (TTFT and TPS) ---
    print("\n[Phase 2: Inference (Use Case Queries)]")
    
    all_ttft_ms = []
    all_tps = []

    for query in TEST_QUERIES:
        print(f"\nQuery: {query[:40]}...")
        
        llm.reset_timings() # Reset internal timers
        
        # Run the inference
        output = llm(
            query,
            max_tokens=256, # Limit output for benchmarking
            echo=False      # Don't print the prompt
        )
        
        # Get timings from the llama.cpp backend
        timings = llm.timings()
        
        # TTFT: Time to process the prompt and generate the *first* token
        # This is the most crucial metric for user-perceived speed.
        ttft_ms = timings.t_prompt_ms
        all_ttft_ms.append(ttft_ms)

        # TPS: Tokens per second for *generation* (not prompt processing)
        # This measures the "streaming" speed.
        tps = 0
        if timings.t_eval_ms > 0:
            # n_eval = number of tokens generated
            # t_eval_ms = time to generate those tokens
            tps = timings.n_eval / (timings.t_eval_ms / 1000.0)
        
        all_tps.append(tps)

        print(f"  -> TTFT (Prompt eval): {ttft_ms:.2f} ms")
        print(f"  -> TPS (Generation): {tps:.2f} tokens/sec")
        print(f"  -> Tokens Generated: {timings.n_eval}")
        print(f"  -> Output snippet: {output['choices'][0]['text'][:50]}...")

    # --- 3. Final Report ---
    print("\n[Phase 3: Final Report]")
    print(f"Model: {os.path.basename(MODEL_PATH)}")
    print(f"Threads: {LLM_PARAMS.get('n_threads')}")
    print("-" * 30)
    print(f"Model RAM Footprint: {model_ram_usage:.2f} MB")
    print(f"Avg. TTFT (Prompt Eval): {sum(all_ttft_ms) / len(all_ttft_ms):.2f} ms")
    print(f"Avg. TPS (Generation): {sum(all_tps) / len(all_tps):.2f} tokens/sec")
    print("--- Benchmark Complete ---")


if __name__ == "__main__":
    run_benchmark()