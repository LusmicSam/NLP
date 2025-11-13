# LLM Edge Device Simulation Environment

This project uses Docker and `llama.cpp` to simulate a resource-constrained (low-CPU, low-RAM) mobile device environment on your laptop. It allows you to benchmark the performance of various GGUF-quantized Language Models for CPU-only inference.

## Files in this Project

1.  **`run_test.sh` / `run_test.bat`**: Your main **configuration script**. You edit this file to set the simulated CPU/RAM limits.
2.  **`Dockerfile`**: Defines the container, installing Python and `llama.cpp`.
3.  **`benchmark.py`**: The Python script that runs the *actual* tests (RAM usage, TTFT, TPS).
4.  **`README.md`**: This file.

---

## 🏃‍♂️ How to Run (New Workflow)

This workflow is much simpler. You only need to run **one command**.

### Step 1: Prerequisites

1.  **Install Docker:** You must have [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine on Linux) installed and running.
2.  **Create Model Folder:** Create a folder named `llm-models` in the *same directory* as these other files.
3.  **Download GGUF Models:** Go to [Hugging Face](https://huggingface.co) and download quantized GGUF models (e.g., `phi-3-mini-4k-instruct-q4_K_M.gguf`). Place them *inside* your `llm-models` folder.

### Step 2: Configure Your Test

You only need to edit **two files** to set up a new test:

1.  **Configure Hardware (in `run_test.sh` or `run_test.bat`)**
    Open your `run_test` script and edit the variables at the top:
    ```bash
    # Set the simulated hardware limits
    SIM_CPU_CORES="2.0"  # Set to "4.0" for a mid-range phone
    SIM_MEMORY="3g"      # Set to "6g" for a mid-range phone
    ```

2.  **Configure Model (in `benchmark.py`)**
    Open `benchmark.py` and edit the variables at the top:
    ```python
    # Set the model file you want to test
    MODEL_PATH = "/models/phi-3-mini-4k-instruct-q4_K_M.gguf"
    
    # ...
    
    # Set n_threads to match your SIM_CPU_CORES (e.g., "2.0" -> 2)
    LLM_PARAMS = {
        "n_threads": 2, 
        "n_gpu_layers": 0 
    }
    ```

### Step 3: Run the Benchmark

Now, just run the *single script* from your terminal.

**On Linux/macOS:**
*First, make the script executable (only need to do this once):*
```bash
chmod +x run_test.sh