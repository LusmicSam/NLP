import torch
import torch.nn.utils.prune as prune

def apply_structured_pruning(model, pruning_amount=0.3):
    """Apply structured pruning to linear layers"""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.ln_structured(module, name='weight', amount=pruning_amount, n=1, dim=0)
            prune.remove(module, 'weight')
    return model

def apply_magnitude_pruning(model, pruning_amount=0.3):
    """Apply magnitude-based pruning"""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_amount)
            prune.remove(module, 'weight')
    return model
