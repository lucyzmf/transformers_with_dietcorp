# %%
from pathlib import Path
import seaborn as sns
import pandas as pd
import os
import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from typing import Callable, Dict, List, Tuple
from tqdm import tqdm
from neural_decoder.inference_funcs import load_bit_phoneme_model, evaluate_model
from neural_decoder.dataset import getDatasetLoaders

data_file = '/home/uchiha/research/transformers_with_dietcorp/processed_data/card_data'
trainLoaders, testLoaders, loadedData = getDatasetLoaders(
        data_file, 8, None,
        False
)


# %%

def count_parameters(model: nn.Module) -> int:
    """Count total number of parameters in model."""
    return sum(p.numel() for p in model.parameters())


def count_nonzero_parameters(model: nn.Module) -> int:
    """Count number of non-zero parameters in model."""
    non_zero_count = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.Conv1d, nn.Conv3d, nn.GRU, nn.LSTM, nn.RNN,
                               nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, 
                               nn.LayerNorm, nn.GroupNorm)):
            state_dict_keys = list(module.state_dict().keys())
            if any("_mask" in k for k in state_dict_keys):
                for n, p in module.state_dict().items():
                    if "_mask" in n:
                        mask = p
                        non_zero_count += torch.count_nonzero(mask).item()
            else:
                count = sum(p.numel() for p in module.parameters())
                non_zero_count += count
        elif isinstance(module, nn.ParameterList):
            for param in module:
                non_zero_count += torch.count_nonzero(param).item()
    return non_zero_count
        


def apply_global_unstructured_pruning(model: nn.Module, amount: float) -> nn.Module:
    """
    Apply global unstructured pruning to all weight parameters.
    
    Args:
        model: PyTorch model to prune
        amount: Fraction of parameters to prune (0.0 to 1.0)
    
    Returns:
        Pruned model
    """
    parameters_to_prune = []
    
    for name, module in model.named_modules():
        # Handle standard layers with 'weight' parameter
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.Conv1d, nn.Conv3d)):
            if hasattr(module, 'weight') and module.weight is not None:
                parameters_to_prune.append((module, 'weight'))
        
        # Handle RNN layers (GRU, LSTM, RNN) - they have weight_ih_l0, weight_hh_l0, etc.
        elif isinstance(module, (nn.GRU, nn.LSTM, nn.RNN)):
            for param_name in dir(module):
                if param_name.startswith('weight_'):
                    param = getattr(module, param_name)
                    if isinstance(param, torch.nn.Parameter):
                        parameters_to_prune.append((module, param_name))
        
        # Handle ParameterList - iterate through each parameter
        elif isinstance(module, nn.ParameterList):
            for idx, param in enumerate(module):
                if param is not None:
                    # We can't directly prune ParameterList items with the standard API
                    # Instead we'll handle these separately after collecting all params
                    pass
        
        # Handle other common layer types
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, 
                                 nn.LayerNorm, nn.GroupNorm)):
            if hasattr(module, 'weight') and module.weight is not None:
                parameters_to_prune.append((module, 'weight'))
    
    
    # Collect all parameters including those in ParameterLists
    all_params_for_global = []
    param_info = []  # Store (module, param_name) or (param_list, idx) info
    
    for name, module in model.named_modules():
        if isinstance(module, nn.ParameterList):
            for idx, param in enumerate(module):
                if param is not None:
                    all_params_for_global.append(param)
                    param_info.append(('paramlist', module, idx))
    
    # Add standard prunable parameters
    for module, param_name in parameters_to_prune:
        param = getattr(module, param_name)
        all_params_for_global.append(param)
        param_info.append(('standard', module, param_name))
    
    if len(all_params_for_global) == 0:
        print("Warning: No parameters found to prune. Check your model architecture.")
        return model
    
    # Calculate global threshold
    all_weights = torch.cat([p.data.abs().flatten() for p in all_params_for_global])
    threshold = torch.kthvalue(all_weights, int(amount * all_weights.numel()))[0]
    
    # Apply pruning mask to each parameter
    for param, info in zip(all_params_for_global, param_info):
        # Compute threshold for this parameter
        # threshold = torch.kthvalue(param.data.abs().flatten(), int(amount * param.data.numel()))[0]
        mask = (param.data.abs() >= threshold).float()
        
        if info[0] == 'paramlist':
            # Directly apply mask to ParameterList parameters
            param.data *= mask
        else:
            # Use standard pruning API for regular modules
            module, param_name = info[1], info[2]
            prune.custom_from_mask(module, param_name, mask)
    
    return model

def evaluate_pruning_impact(
    model: nn.Module,
    prune_percentages: List[float] = None,
) -> Tuple[List[float], List[int], List[float], List[float]]:
    """
    Evaluate model performance at different pruning levels.
    
    Args:
        model: Original PyTorch model
        val_loader: Validation data loader
        prune_percentages: List of pruning percentages to test (0-100)
        device: Device to run evaluation on
        criterion: Loss function
    
    Returns:
        Tuple of (prune_percentages, param_counts, accuracies, losses)
    """
    if prune_percentages is None:
        prune_percentages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 99]
    
    total_params = count_parameters(model)
    
    results = {
        'prune_pct': [],
        'param_count': [],
        'cer': [],
    }
    
    print(f"Total parameters in model: {total_params:,}")
    print(f"Evaluating {len(prune_percentages)} pruning levels...\n")
    
    for pct in tqdm(prune_percentages, desc="Pruning evaluation"):
        # Create a fresh copy of the model for each pruning level
        model_copy = deepcopy(model)
        
        if pct > 0:
            # Apply pruning
            model_copy = apply_global_unstructured_pruning(model_copy, pct / 100.0)
        
        # Evaluate
        _, metrics, _ = evaluate_model(model_copy, loadedData, args, partition='test', device='cuda')
        
        # Count remaining parameters
        remaining_params = count_nonzero_parameters(model_copy)
        
        results['prune_pct'].append(pct)
        results['param_count'].append(remaining_params)
        results['cer'].append(metrics)
        
        print(f"Pruned {pct}%: {remaining_params:,} params remaining, "
              f"CER: {metrics:.2f}%")
        
        # Clean up
        del model_copy
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return (results['prune_pct'], results['param_count'], 
            results['cer'])

# %%
def plot_pruning_results(
    prune_percentages: List[float],
    param_counts: List[int],
    cers: List[float],
    save_path: str = 'pruning_results.png'
):
    """
    Plot pruning results with dual x-axis (percentage and param count).
    
    Args:
        prune_percentages: List of pruning percentages
        param_counts: List of remaining parameter counts
        cers: List of CERs
        save_path: Path to save the plot
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
    
    # Plot: CER vs Pruning Percentage
    ax1.plot(prune_percentages, cers, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Pruning Percentage (%)', fontsize=12)
    ax1.set_ylabel('Character Error Rate (%)', fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Model Performance vs Pruning Level', fontsize=14, fontweight='bold')
    
    # Secondary x-axis for parameter count
    ax1_top = ax1.twiny()
    # Map pruning percentages to parameter counts for the secondary axis
    ax1_top.set_xlim(ax1.get_xlim())
    # Create tick positions that align with the primary axis
    param_count_ticks = np.linspace(min(param_counts), max(param_counts), len(prune_percentages))
    ax1_top.set_xticks(prune_percentages)
    ax1_top.set_xticklabels([f'{int(pc/1e6)}M' if pc >= 1e6 else f'{int(pc/1000)}K' 
                              for pc in param_counts])
    ax1_top.set_xlabel('Remaining Parameters', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")
    plt.show()


# %%
if __name__ == "__main__": 
    
    model_size = "large"
    model_name = f"time_masked_{model_size}"
    bit_phone_filepath = Path("/home/uchiha/research/transformers_with_dietcorp/outputs") / f"{model_name}_seed_1"
    model, args = load_bit_phoneme_model(bit_phone_filepath)
    device = 'cuda'
    model = model.to(device)
    
        # Run pruning evaluation
    prune_pcts, param_counts, accs = evaluate_pruning_impact(
        model=model,
        prune_percentages=[0, 50, 75, 85, 90, 95, 98, 99],
    )
    
    # %%
    # Plot results
    # plot_pruning_results(prune_pcts, param_counts, accs, )
    df = pd.DataFrame({
        'Prune Percentage (%)': prune_pcts,
        'Remaining Parameters (M)': [c/1e6 for c in param_counts],
        'Character Error Rate': accs,
    })
    
    plt.figure(figsize=(12, 5))
    sns.lineplot(x='Remaining Parameters (M)' ,y='Character Error Rate', data=df, marker='o')
    
    # add annot
    for i in range(len(df)):
        plt.text(df['Remaining Parameters (M)'][i], df['Character Error Rate'][i],
                 f"{df['Character Error Rate'][i]:.3f}",
                 fontsize=10, ha='center', va='bottom')
    plt.title('Model Performance vs Pruning Level 32M transformer', fontsize=14, fontweight='bold')
    plt.show()
    
    # %%
    # check prune percentage per param group 
    pruned_model = apply_global_unstructured_pruning(deepcopy(model), amount=0.5)
    record = []
    for name, module in pruned_model.named_modules():
        try:
            layer_type = name.split('.')[4]
        except:
            if "to_patch_embedding" in name:
                layer_type = "to_patch_embedding"
            else:
                layer_type = name
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.Conv1d, nn.Conv3d, nn.GRU, nn.LSTM, nn.RNN,
                               nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, 
                               nn.LayerNorm, nn.GroupNorm)):
            for n, p in module.state_dict().items():
                if "_mask" in n:
                    mask = p
                    total_params = mask.nelement()
                    nonzero_params = torch.count_nonzero(mask).item()
                    percent_pruned = 100.0 * (total_params - nonzero_params) / total_params
                    record.append((f"{name}.{n}", percent_pruned, layer_type))
        elif isinstance(module, nn.ParameterList):
            for idx, param in enumerate(module):
                total_params = param.nelement()
                nonzero_params = torch.count_nonzero(param).item()
                percent_pruned = 100.0 * (total_params - nonzero_params) / total_params
                record.append((f"{name}[{idx}]", percent_pruned, "day_layer"))
    
    df = pd.DataFrame.from_records(record, columns=['Parameter Group', 'Percent Pruned', 'Layer Type'])
    print(df)
    
    # %%
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Parameter Group', y='Percent Pruned', data=df, hue='Layer Type', )
    plt.xticks(rotation=90)
    plt.ylabel('Percentage of Parameters Pruned (%)')
    plt.title('Pruning Percentage per Parameter Group at 50% Global Pruning', fontsize=14, fontweight='bold')
    plt.legend(title='Layer Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    

    # %% multiple checkpoints
    model_name_size_map = {
        'large': '32.8M',
        'medium': '12.9M',
        'small': '3.4M',
    }
    base_path = Path("/home/uchiha/research/transformers_with_dietcorp/outputs")
    df_all = []
    
    for model_size in ['small', 'medium', 'large']:
        model_name = f"time_masked_{model_size}"
        model_path = base_path / f"{model_name}_seed_1"

        print(f"Evaluating model at: {model_path}")
        model, args = load_bit_phoneme_model(model_path)
        device = 'cuda'
        model = model.to(device)

        prune_pcts, param_counts, accs = evaluate_pruning_impact(
            model=model,
            prune_percentages=[0, 50, 75, 85, 90, 95, 98, 99],
        )
        
        df = pd.DataFrame({
            'Prune Percentage (%)': prune_pcts,
            'Remaining Parameters (M)': [c/1e6 for c in param_counts],
            'Character Error Rate': accs,
            'Model': model_size + f" ({model_name_size_map[model_size]})",
        })
        df_all.append(df)
    
    df_combined = pd.concat(df_all, ignore_index=True)
    # %%
    plt.figure(figsize=(12, 5))
    sns.lineplot(x='Remaining Parameters (M)' ,y='Character Error Rate', data=df_combined, marker='o', hue='Model')     
    # annotate
    for i in range(len(df_combined)):
        plt.text(df_combined['Remaining Parameters (M)'][i], df_combined['Character Error Rate'][i],
                 f"{df_combined['Character Error Rate'][i]:.3f}",
                 fontsize=8, ha='center', va='bottom')
    plt.legend(title='Model Size')
    plt.title('Model Performance vs Pruning Level', fontsize=14, fontweight='bold')
    # plt.ylim(0, 1)
    xmin, xmax = plt.xlim()
    plt.hlines(y=1.0, xmin=xmin, xmax=xmax, colors='r', linestyles='dashed', label='CER=1.0')
    plt.show()

    
        

# %%
