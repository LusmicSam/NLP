import json
import pandas as pd

def generate_recommendation_rules(results_file='benchmark_results.json'):
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    success_df = df[df['status'] == 'success']
    
    # Create rules based on device constraints
    rules = {}
    for device in success_df['device'].unique():
        device_data = success_df[success_df['device'] == device]
        rules[device] = {
            "max_model_size": device_data['parameters'].max(),
            "best_latency_model": device_data.loc[device_data['avg_latency_ms'].idxmin()]['model'],
            "best_accuracy_model": device_data.loc[device_data['accuracy'].idxmax()]['model']
        }
    
    with open('recommendation_rules.json', 'w') as f:
        json.dump(rules, f, indent=2)
    
    print("Recommendation rules generated!")

if __name__ == "__main__":
    generate_recommendation_rules()
