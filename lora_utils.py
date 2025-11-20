from peft import LoraConfig, get_peft_model

def apply_lora(model, r=16, alpha=32):
    """Apply LoRA to model"""
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    lora_model = get_peft_model(model, config)
    return lora_model
