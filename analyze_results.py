import json
import pandas as pd

def load_and_analyze(json_file='benchmark_results.json'):
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    success_df = df[df['status'] == 'success']
    
    # Pivot table: Model vs Device vs Quantization
    pivot = success_df.pivot_table(
        values=['avg_latency_ms', 'peak_memory_gb', 'tokens_per_second'],
        index=['model', 'device'],
        columns=['quantization'],
        aggfunc='mean'
    )
    
    print("Benchmark Summary:")
    print(pivot)
    
    # Save to CSV
    success_df.to_csv('benchmark_summary.csv', index=False)
    print("\nSaved detailed results to benchmark_summary.csv")

if __name__ == "__main__":
    load_and_analyze()
