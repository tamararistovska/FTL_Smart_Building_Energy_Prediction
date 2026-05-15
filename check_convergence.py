"""
Convergence Analysis: Train vs Validation Loss
Check whether to train longer or if models are converged
"""

# ============= PASTE YOUR MLP_LOSS_CURVES HERE =============
MLP_LOSS_CURVES = [
    {"strategy": "FTL", "train_loss": [1.2, 1.05, 0.94, 0.87, 0.81, 0.76, 0.72, 0.69, 0.66, 0.64]},
    {"strategy": "FTL", "train_loss": [1.25, 1.08, 0.97, 0.89, 0.83, 0.78, 0.74, 0.71, 0.68, 0.65]},
    {"strategy": "Personalized-FL", "train_loss": [0.95, 0.8, 0.71, 0.65, 0.6, 0.56, 0.53, 0.51, 0.49, 0.47]},
    {"strategy": "Personalized-FL", "train_loss": [0.98, 0.83, 0.73, 0.67, 0.62, 0.58, 0.55, 0.52, 0.5, 0.48]},
    {"strategy": "Progressive Unfreezing", "train_loss": [2.95, 2.78, 2.61, 2.46, 2.33, 2.21, 2.1, 2.01, 1.93, 1.86]},
    {"strategy": "Progressive Unfreezing", "train_loss": [3.5, 3.25, 3.02, 2.82, 2.64, 2.48, 2.34, 2.22, 2.11, 2.02]},
    {"strategy": "Instance-TL", "train_loss": [2.6, 2.43, 2.29, 2.16, 2.04, 1.94, 1.85, 1.77, 1.7, 1.64]},
    {"strategy": "Instance-TL", "train_loss": [2.65, 2.48, 2.33, 2.19, 2.07, 1.97, 1.87, 1.79, 1.72, 1.65]},
    {"strategy": "Fed-SimTL", "train_loss": [2.35, 2.19, 2.05, 1.93, 1.82, 1.72, 1.63, 1.56, 1.49, 1.43]},
    {"strategy": "Fed-SimTL", "train_loss": [2.4, 2.24, 2.09, 1.96, 1.85, 1.75, 1.66, 1.58, 1.52, 1.45]},
    {"strategy": "FedMetaTL", "train_loss": [2.38, 2.22, 2.08, 1.96, 1.85, 1.75, 1.66, 1.59, 1.52, 1.46]},
    {"strategy": "FedMetaTL", "train_loss": [2.42, 2.26, 2.11, 1.99, 1.88, 1.78, 1.69, 1.61, 1.54, 1.48]},
]
# =========================================================

import numpy as np
from collections import defaultdict

print("\n" + "="*70)
print("CONVERGENCE ANALYSIS: When to Stop Training?")
print("="*70)

# Check if validation loss available
has_val_loss = any('val_loss' in entry for entry in MLP_LOSS_CURVES)

if not has_val_loss:
    print("\n⚠️  NO VALIDATION LOSS FOUND")
    print("\nWithout validation loss, we'll analyze training loss convergence.")
    print("Recommendation: Check if validation metrics (MAE, RMSE) on test set")
    print("               are still improving.\n")
    
    # Analyze training loss convergence
    grouped = defaultdict(list)
    for entry in MLP_LOSS_CURVES:
        strat = entry.get('strategy', 'Unknown')
        grouped[strat].append(np.array(entry['train_loss']))
    
    print("TRAINING LOSS CONVERGENCE ANALYSIS:")
    print("-" * 70)
    print(f"{'Strategy':<25} {'Final Loss':>12} {'Improvement':>15} {'Recommendation':>18}")
    print("-" * 70)
    
    for strat, curves in sorted(grouped.items()):
        curves = np.array(curves)
        final_loss = curves[:, -1].mean()
        initial_loss = curves[:, 0].mean()
        improvement_last_3 = (curves[:, -3].mean() - curves[:, -1].mean())
        total_improvement = (initial_loss - final_loss) / initial_loss * 100
        
        # Decision logic
        if improvement_last_3 > 0.05:  # > 0.05 loss units in last 3 epochs
            recommendation = "Train longer ↑"
        elif improvement_last_3 > 0.01:
            recommendation = "Maybe 5-10 more"
        else:
            recommendation = "Converged ✓"
        
        print(f"{strat:<25} {final_loss:>12.4f} {improvement_last_3:>+15.4f} {recommendation:>18}")
    
    print("\n" + "-" * 70)
    print("DECISION MATRIX (Last 3 epochs improvement):")
    print("  > 0.05:  Still learning meaningfully  → Train 15-20 epochs")
    print("  0.01-0.05: Diminishing returns      → Train 12-15 epochs")
    print("  < 0.01:  Nearly converged           → Stop at current")
    
    print("\n" + "="*70)
    print("SUMMARY RECOMMENDATION:")
    print("-" * 70)
    print("\nBased on training loss alone:")
    print("\n✓ LIKELY CONVERGED (10-12 epochs OK):")
    for strat, curves in sorted(grouped.items()):
        curves = np.array(curves)
        improvement_last_3 = (curves[:, -3].mean() - curves[:, -1].mean())
        if improvement_last_3 <= 0.01:
            print(f"    • {strat}")
    
    print("\n↑ TRAIN LONGER (15-20 epochs recommended):")
    for strat, curves in sorted(grouped.items()):
        curves = np.array(curves)
        improvement_last_3 = (curves[:, -3].mean() - curves[:, -1].mean())
        if improvement_last_3 > 0.05:
            print(f"    • {strat}")
    
else:
    print("\n✓ VALIDATION LOSS DATA FOUND!\n")
    
    grouped = defaultdict(list)
    for entry in MLP_LOSS_CURVES:
        strat = entry.get('strategy', 'Unknown')
        grouped[strat].append({
            'train': np.array(entry['train_loss']),
            'val': np.array(entry['val_loss'])
        })
    
    print("OVERFITTING RISK ANALYSIS:")
    print("-" * 70)
    print(f"{'Strategy':<25} {'Risk':>12} {'Val Gap':>12} {'Trend':>12}")
    print("-" * 70)
    
    for strat, curves in sorted(grouped.items()):
        trains = np.array([c['train'] for c in curves])
        vals = np.array([c['val'] for c in curves])
        
        train_final = trains[:, -1].mean()
        val_final = vals[:, -1].mean()
        gap = val_final - train_final
        gap_pct = (gap / train_final * 100) if train_final > 0 else 0
        
        # Risk assessment
        if gap_pct > 15:
            risk = "🔴 HIGH"
        elif gap_pct > 5:
            risk = "🟡 MEDIUM"
        else:
            risk = "🟢 LOW"
        
        # Trend
        gap_start = (vals[:, 0] - trains[:, 0]).mean()
        if gap > gap_start * 1.1:
            trend = "Getting worse ↗"
        elif gap > gap_start * 0.9:
            trend = "Stable →"
        else:
            trend = "Improving ↘"
        
        print(f"{strat:<25} {risk:>12} {gap:>+12.4f} {trend:>12}")
    
    print("\n" + "="*70)

print("\n")
