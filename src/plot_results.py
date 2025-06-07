# src/plot_results_with_statistics.py

import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from model import MLP

# ── Réglages globaux Matplotlib ─────────────────────────────────────────────
plt.rcParams.update({"font.size": 10})

def calculate_improvement_percentage(baseline, improved):
    """Calculate percentage improvement from baseline to improved value"""
    return ((baseline - improved) / baseline) * 100

def print_quantified_gains(spars0_pct, spars1_pct, pl0_stats, pl1_stats):
    """Print quantified gains for discussion section"""
    print("\n" + "="*60)
    print("GAINS QUANTIFIÉS POUR LA DISCUSSION")
    print("="*60)
    
    # Find closest sparsity levels for comparison
    target_sparsities = [5, 10, 20]  # Target sparsity percentages for comparison
    
    for target_spar in target_sparsities:
        # Find closest PL0 configuration
        pl0_idx = min(range(len(spars0_pct)), key=lambda i: abs(spars0_pct[i] - target_spar))
        pl0_spar_actual = spars0_pct[pl0_idx]
        
        # Find closest PL1 configuration  
        pl1_idx = min(range(len(spars1_pct)), key=lambda i: abs(spars1_pct[i] - target_spar))
        pl1_spar_actual = spars1_pct[pl1_idx]
        
        if abs(pl0_spar_actual - target_spar) < 3 and abs(pl1_spar_actual - target_spar) < 3:
            # Get the corresponding k and lambda values
            k_val = sorted(pl0_stats.keys())[pl0_idx]
            lam_val = sorted(pl1_stats.keys())[pl1_idx]
            
            # Calculate improvements (assuming dense model has 0% sparsity)
            # For baseline, use the configuration with lowest sparsity
            baseline_k = sorted(pl0_stats.keys())[0]  # Lowest sparsity PL0
            baseline_lam = sorted(pl1_stats.keys())[0]  # Lowest sparsity PL1
            
            print(f"\n--- À ~{target_spar:.0f}% de sparsité ---")
            
            # PL0 gains
            lat_baseline = pl0_stats[baseline_k]['latency_mean']
            lat_sparse = pl0_stats[k_val]['latency_mean']
            lat_gain = calculate_improvement_percentage(lat_baseline, lat_sparse)
            
            energy_baseline = pl0_stats[baseline_k]['energy_mean']
            energy_sparse = pl0_stats[k_val]['energy_mean']
            energy_gain = calculate_improvement_percentage(energy_baseline, energy_sparse)
            
            acc_baseline = pl0_stats[baseline_k]['acc_mean']
            acc_sparse = pl0_stats[k_val]['acc_mean']
            acc_loss = abs(calculate_improvement_percentage(acc_baseline, acc_sparse))
            
            print(f"PL0 (k={k_val}):")
            print(f"  • Latence: {lat_baseline:.3f} → {lat_sparse:.3f} ms ({lat_gain:+.1f}%)")
            print(f"  • Énergie: {energy_baseline:.2f} → {energy_sparse:.2f} mJ ({energy_gain:+.1f}%)")
            print(f"  • Accuracy: {acc_baseline:.4f} → {acc_sparse:.4f} (-{acc_loss:.2f}%)")
            
            # PL1 gains
            lat_baseline = pl1_stats[baseline_lam]['latency_mean']
            lat_sparse = pl1_stats[lam_val]['latency_mean']
            lat_gain = calculate_improvement_percentage(lat_baseline, lat_sparse)
            
            energy_baseline = pl1_stats[baseline_lam]['energy_mean']
            energy_sparse = pl1_stats[lam_val]['energy_mean']
            energy_gain = calculate_improvement_percentage(energy_baseline, energy_sparse)
            
            acc_baseline = pl1_stats[baseline_lam]['acc_mean']
            acc_sparse = pl1_stats[lam_val]['acc_mean']
            acc_loss = abs(calculate_improvement_percentage(acc_baseline, acc_sparse))
            
            print(f"PL1 (λ={lam_val:.0e}):")
            print(f"  • Latence: {lat_baseline:.3f} → {lat_sparse:.3f} ms ({lat_gain:+.1f}%)")
            print(f"  • Énergie: {energy_baseline:.2f} → {energy_sparse:.2f} mJ ({energy_gain:+.1f}%)")
            print(f"  • Accuracy: {acc_baseline:.4f} → {acc_sparse:.4f} (-{acc_loss:.2f}%)")

def plot_temporal_evolution(pl0_stats, pl1_stats):
    """Generate temporal evolution plots for accuracy and energy vs epochs"""
    
    # Find representative configurations (moderate sparsity)
    # For PL0, select k that gives around 10-15% sparsity
    dummy = MLP()
    TOTAL_W = sum(p.numel() for p in dummy.parameters())
    
    # Select configurations that give moderate sparsity for better visualization
    target_sparsity = 0.10  # Target 10% sparsity
    
    # Find best PL0 configuration
    best_pl0_k = None
    best_pl0_diff = float('inf')
    for k in pl0_stats.keys():
        sparsity_mean = pl0_stats[k]['sparsity_mean']
        diff = abs(sparsity_mean - target_sparsity)
        if diff < best_pl0_diff:
            best_pl0_diff = diff
            best_pl0_k = k
    
    # Find best PL1 configuration
    best_pl1_lam = None
    best_pl1_diff = float('inf')
    for lam in pl1_stats.keys():
        sparsity_mean = pl1_stats[lam]['sparsity_mean']
        diff = abs(sparsity_mean - target_sparsity)
        if diff < best_pl1_diff:
            best_pl1_diff = diff
            best_pl1_lam = lam
    
    if best_pl0_k is None or best_pl1_lam is None:
        print("Attention: Impossible de trouver des configurations appropriées pour l'évolution temporelle")
        return
    
    # Get training histories
    pl0_all_runs = pl0_stats[best_pl0_k]['all_runs']
    pl1_all_runs = pl1_stats[best_pl1_lam]['all_runs']
    
    # Calculate mean and std across runs for each epoch
    epochs = len(pl0_all_runs[0]['training_history']['acc'])
    epoch_numbers = list(range(1, epochs + 1))
    
    # PL0 statistics across epochs
    pl0_acc_epochs = []
    pl0_energy_epochs = []
    
    for epoch in range(epochs):
        epoch_accs = [run['training_history']['acc'][epoch] for run in pl0_all_runs]
        epoch_energies = [run['energy_mj'] for run in pl0_all_runs]  # Energy is constant per run
        
        pl0_acc_epochs.append({
            'mean': np.mean(epoch_accs),
            'std': np.std(epoch_accs)
        })
        pl0_energy_epochs.append({
            'mean': np.mean(epoch_energies),
            'std': np.std(epoch_energies)
        })
    
    # PL1 statistics across epochs
    pl1_acc_epochs = []
    pl1_energy_epochs = []
    
    for epoch in range(epochs):
        epoch_accs = [run['training_history']['acc'][epoch] for run in pl1_all_runs]
        epoch_energies = [run['energy_mj'] for run in pl1_all_runs]  # Energy is constant per run
        
        pl1_acc_epochs.append({
            'mean': np.mean(epoch_accs),
            'std': np.std(epoch_accs)
        })
        pl1_energy_epochs.append({
            'mean': np.mean(epoch_energies),
            'std': np.std(epoch_energies)
        })
    
    # Plot Accuracy vs Epoch
    plt.figure(figsize=(6, 5))
    
    pl0_acc_means = [epoch['mean'] for epoch in pl0_acc_epochs]
    pl0_acc_stds = [epoch['std'] for epoch in pl0_acc_epochs]
    pl1_acc_means = [epoch['mean'] for epoch in pl1_acc_epochs]
    pl1_acc_stds = [epoch['std'] for epoch in pl1_acc_epochs]
    
    plt.errorbar(epoch_numbers, pl0_acc_means, yerr=pl0_acc_stds, 
                 marker='x', linestyle='-', label=f"P-L₀ (k={best_pl0_k})", 
                 capsize=3, capthick=1, color='blue')
    plt.errorbar(epoch_numbers, pl1_acc_means, yerr=pl1_acc_stds, 
                 marker='o', linestyle='-', label=f"P-L₁ (λ={best_pl1_lam:.0e})", 
                 capsize=3, capthick=1, color='orange')
    
    plt.xlabel("Époque")
    plt.ylabel("Accuracy (test)")
    plt.title("Évolution de l'Accuracy durant l'entraînement")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("accuracy_vs_epoch.png", dpi=300)
    plt.show()
    
    # Plot Energy vs Epoch (constant per configuration but shown for consistency)
    plt.figure(figsize=(6, 5))
    
    pl0_energy_means = [epoch['mean'] for epoch in pl0_energy_epochs]
    pl0_energy_stds = [epoch['std'] for epoch in pl0_energy_epochs]
    pl1_energy_means = [epoch['mean'] for epoch in pl1_energy_epochs]
    pl1_energy_stds = [epoch['std'] for epoch in pl1_energy_epochs]
    
    plt.errorbar(epoch_numbers, pl0_energy_means, yerr=pl0_energy_stds, 
                 marker='x', linestyle='-', label=f"P-L₀ (k={best_pl0_k})", 
                 capsize=3, capthick=1, color='blue')
    plt.errorbar(epoch_numbers, pl1_energy_means, yerr=pl1_energy_stds, 
                 marker='o', linestyle='-', label=f"P-L₁ (λ={best_pl1_lam:.0e})", 
                 capsize=3, capthick=1, color='orange')
    
    plt.xlabel("Époque")
    plt.ylabel("Énergie (mJ)")
    plt.title("Coût énergétique estimé par configuration")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("energy_vs_epoch.png", dpi=300)
    plt.show()
    
    print(f"\n✅ Graphiques d'évolution temporelle générés:")
    print(f"   • PL0: k={best_pl0_k} (sparsité: {pl0_stats[best_pl0_k]['sparsity_mean']:.3f})")
    print(f"   • PL1: λ={best_pl1_lam:.0e} (sparsité: {pl1_stats[best_pl1_lam]['sparsity_mean']:.3f})")

def main():
    # ── 1. Charger les résultats avec statistiques ─────────────────────────────
    try:
        with open("results_PL0_stats.pkl", "rb") as f:
            pl0_stats = pickle.load(f)
        with open("results_PL1_stats.pkl", "rb") as f:
            pl1_stats = pickle.load(f)
    except FileNotFoundError:
        print("Erreur: Fichiers de résultats statistiques non trouvés.")
        print("Veuillez d'abord exécuter train_pl0_multiple_runs.py et train_pl1_multiple_runs.py")
        return

    # ── 2. Calculer TOTAL_W pour passer en % de sparsité ──────────────────────
    dummy = MLP()
    TOTAL_W = sum(p.numel() for p in dummy.parameters())

    # Extract sparsity percentages
    spars0 = sorted(pl0_stats.keys())
    spars1 = sorted(pl1_stats.keys())
    spars0_pct = [k / TOTAL_W * 100 for k in spars0]
    spars1_pct = [lam / TOTAL_W * 100 for lam in spars1]

    # Extract means and stds for plotting
    acc0_mean = [pl0_stats[k]['acc_mean'] for k in spars0]
    acc0_std = [pl0_stats[k]['acc_std'] for k in spars0]
    acc1_mean = [pl1_stats[lam]['acc_mean'] for lam in spars1]
    acc1_std = [pl1_stats[lam]['acc_std'] for lam in spars1]

    lat0_mean = [pl0_stats[k]['latency_mean'] for k in spars0]
    lat0_std = [pl0_stats[k]['latency_std'] for k in spars0]
    lat1_mean = [pl1_stats[lam]['latency_mean'] for lam in spars1]
    lat1_std = [pl1_stats[lam]['latency_std'] for lam in spars1]

    size0_mean = [pl0_stats[k]['size_mean'] for k in spars0]
    size0_std = [pl0_stats[k]['size_std'] for k in spars0]
    size1_mean = [pl1_stats[lam]['size_mean'] for lam in spars1]
    size1_std = [pl1_stats[lam]['size_std'] for lam in spars1]

    en0_mean = [pl0_stats[k]['energy_mean'] for k in spars0]
    en0_std = [pl0_stats[k]['energy_std'] for k in spars0]
    en1_mean = [pl1_stats[lam]['energy_mean'] for lam in spars1]
    en1_std = [pl1_stats[lam]['energy_std'] for lam in spars1]

    # ── F1 : Accuracy vs Sparsité (%) avec barres d'erreur ────────────────────
    plt.figure(figsize=(6, 5))
    plt.errorbar(spars0_pct, acc0_mean, yerr=acc0_std, 
                 marker='x', linestyle='-', label="PL0", capsize=5, capthick=2)
    plt.errorbar(spars1_pct, acc1_mean, yerr=acc1_std, 
                 marker='o', linestyle='-', label="PL1", capsize=5, capthick=2)
    plt.xlabel("Sparsité (%)")
    plt.ylabel("Accuracy (test)")
    plt.title("Compromis Accuracy vs Sparsité")
    plt.ylim(0.94, 1.0)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_acc_vs_sparsity_pct_stats.png", dpi=300)
    plt.show()

    # ── F2 : Latence vs Sparsité (%) avec barres d'erreur ──────────────────────
    plt.figure(figsize=(6, 5))
    plt.errorbar(spars0_pct, lat0_mean, yerr=lat0_std, 
                 marker='x', linestyle='-', label="PL0", capsize=5, capthick=2)
    plt.errorbar(spars1_pct, lat1_mean, yerr=lat1_std, 
                 marker='o', linestyle='-', label="PL1", capsize=5, capthick=2)
    plt.xlabel("Sparsité (%)")
    plt.ylabel("Latence (ms)")
    plt.title("Latence vs Sparsité")
    plt.ylim(0, max(max(lat0_mean), max(lat1_mean)) * 1.2)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_latency_vs_sparsity_pct_stats.png", dpi=300)
    plt.show()

    # ── F3 : Taille modèle vs Sparsité (%) avec barres d'erreur ────────────────
    plt.figure(figsize=(6, 5))
    plt.errorbar(spars0_pct, size0_mean, yerr=size0_std, 
                 marker='x', linestyle='-', label="PL0", capsize=5, capthick=2)
    plt.errorbar(spars1_pct, size1_mean, yerr=size1_std, 
                 marker='o', linestyle='-', label="PL1", capsize=5, capthick=2)
    plt.xlabel("Sparsité (%)")
    plt.ylabel("Taille modèle (MB)")
    plt.title("Taille modèle vs Sparsité")
    all_sizes = size0_mean + size1_mean
    ymin, ymax = min(all_sizes) * 0.95, max(all_sizes) * 1.05
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_size_vs_sparsity_pct_stats.png", dpi=300)
    plt.show()

    # ── F4 : Énergie vs Sparsité (%) avec barres d'erreur ──────────────────────
    plt.figure(figsize=(6, 5))
    plt.errorbar(spars0_pct, en0_mean, yerr=en0_std, 
                 marker='x', linestyle='-', label="PL0", capsize=5, capthick=2)
    plt.errorbar(spars1_pct, en1_mean, yerr=en1_std, 
                 marker='o', linestyle='-', label="PL1", capsize=5, capthick=2)
    plt.xlabel("Sparsité (%)")
    plt.ylabel("Énergie (mJ)")
    plt.title("Énergie vs Sparsité")
    all_energies = en0_mean + en1_mean
    plt.ylim(0, max(all_energies) * 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_energy_vs_sparsity_pct_stats.png", dpi=300)
    plt.show()

    # ── F5 : Graphique combiné Accuracy vs Latence avec barres d'erreur ────────
    plt.figure(figsize=(6, 5))
    plt.errorbar(lat0_mean, acc0_mean, xerr=lat0_std, yerr=acc0_std,
                 marker='x', linestyle='', label="PL0", capsize=5, capthick=2)
    plt.errorbar(lat1_mean, acc1_mean, xerr=lat1_std, yerr=acc1_std,
                 marker='o', linestyle='', label="PL1", capsize=5, capthick=2)
    plt.xlabel("Latence (ms)")
    plt.ylabel("Accuracy (test)")
    plt.title("Compromis Accuracy vs Latence")
    plt.xlim(0, max(max(lat0_mean), max(lat1_mean)) * 1.1)
    plt.ylim(0.94, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_acc_vs_latency_stats.png", dpi=300)
    plt.show()

    # ── F6 : Graphique combiné Accuracy vs Énergie avec barres d'erreur ────────
    plt.figure(figsize=(6, 5))
    plt.errorbar(en0_mean, acc0_mean, xerr=en0_std, yerr=acc0_std,
                 marker='x', linestyle='', label="PL0", capsize=5, capthick=2)
    plt.errorbar(en1_mean, acc1_mean, xerr=en1_std, yerr=acc1_std,
                 marker='o', linestyle='', label="PL1", capsize=5, capthick=2)
    plt.xlabel("Énergie (mJ)")
    plt.ylabel("Accuracy (test)")
    plt.title("Compromis Accuracy vs Énergie")
    plt.xlim(0, max(max(en0_mean), max(en1_mean)) * 1.1)
    plt.ylim(0.94, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_acc_vs_energy_stats.png", dpi=300)
    plt.show()

    # ── F7-F8 : Graphiques d'évolution temporelle ──────────────────────────────
    plot_temporal_evolution(pl0_stats, pl1_stats)

    # ── Calcul et affichage des gains quantifiés ────────────────────────────────
    print_quantified_gains(spars0_pct, spars1_pct, pl0_stats, pl1_stats)

    print("\n✅ Graphiques avec barres d'erreur générés avec succès!")
    print("📊 Fichiers créés:")
    files = [
        "fig_acc_vs_sparsity_pct_stats.png",
        "fig_latency_vs_sparsity_pct_stats.png", 
        "fig_size_vs_sparsity_pct_stats.png",
        "fig_energy_vs_sparsity_pct_stats.png",
        "fig_acc_vs_latency_stats.png",
        "fig_acc_vs_energy_stats.png",
        "accuracy_vs_epoch.png",
        "energy_vs_epoch.png"
    ]
    for file in files:
        print(f"  • {file}")

if __name__ == "__main__":
    main()