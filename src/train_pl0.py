import torch, time, numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from multiprocessing import Pool, cpu_count

from dataset import get_loaders
from model   import MLP
from utils   import accuracy, sparsity, save_pickle, epoch_time
from utils   import benchmark_latency, model_size_mb, estimate_energy_mj
from config  import LR, MOMENTUM, EPOCHS, K_RATIO_GRID

def hard_threshold(param, k):
    flat = param.view(-1)
    if k >= flat.numel():
        return
    thresh = flat.abs().kthvalue(flat.numel() - k).values.item()
    mask = (flat.abs() >= thresh).float()
    param.mul_(mask.view_as(param))

def get_loaders_single():
    """Get data loaders with num_workers=0 for use in multiprocessing workers"""
    train_loader, test_loader = get_loaders()
    # Force single process loading
    train_loader = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_loader.dataset,
        batch_size=test_loader.batch_size,
        shuffle=False,
        num_workers=0
    )
    return train_loader, test_loader

def train_single_run(args):
    """Train a single model with given k and return final metrics"""
    k, device, sample_input, seed, run_num = args
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Create data loaders inside the worker process with num_workers=0
    train_loader, test_loader = get_loaders_single()
    
    model = MLP().to(device)
    optim_ = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    criterion = nn.CrossEntropyLoss()
    
    # Store metrics for each epoch (for temporal evolution plots)
    training_history = {'loss': [], 'acc': [], 'sparsity': [], 'time': []}
    
    print(f"  Starting Run {run_num} for k={k}")
    
    for epoch in trange(EPOCHS, desc=f"Run {run_num}, k={k}"):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim_.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optim_.step()

            # Apply hard thresholding after each gradient step
            with torch.no_grad():
                for p in model.parameters():
                    hard_threshold(p, k)

            running_loss += loss.item() * y.size(0)

        # Calculate epoch metrics
        epoch_duration = epoch_time(t0)
        avg_loss = running_loss / len(train_loader.dataset)
        test_acc = accuracy(model, test_loader, device)
        sprs = sparsity(model)

        # Store metrics for this epoch
        training_history['loss'].append(avg_loss)
        training_history['acc'].append(test_acc)
        training_history['sparsity'].append(sprs)
        training_history['time'].append(epoch_duration)

        # Print epoch progress
        if epoch % 10 == 0 or epoch == EPOCHS - 1:  # Print every 10 epochs and last epoch
            print(f"    [Run {run_num}, k={k}] Epoch {epoch+1}/{EPOCHS} "
                  f"Loss={avg_loss:.4f} Acc={test_acc:.4f} "
                  f"Sparsity={sprs:.3f} Time={epoch_duration:.1f}s")

    # Calculate final metrics after training
    final_test_acc = accuracy(model, test_loader, device)
    final_sparsity = sparsity(model)
    lat_ms = benchmark_latency(model, sample_input, device)
    size_mb = model_size_mb(model)
    energy_mj = estimate_energy_mj(lat_ms)
    
    # Prepare results dictionary
    run_results = {
        'final_acc': final_test_acc,
        'final_sparsity': final_sparsity,
        'latency_ms': lat_ms,
        'size_mb': size_mb,
        'energy_mj': energy_mj,
        'training_history': training_history  # This is what plot_temporal_evolution() needs
    }
    
    print(f"  ✅ Run {run_num} completed: Acc={final_test_acc:.4f}, "
          f"Sparsity={final_sparsity:.3f}, Latency={lat_ms:.2f}ms, Energy={energy_mj:.2f}mJ")
    
    return run_results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare a single input example for latency benchmarking
    train_loader, _ = get_loaders_single()  # Use single process loader here too
    sample_batch, _ = next(iter(train_loader))
    sample_input = sample_batch[:1].to(device)

    # Compute total weights to build K_GRID
    dummy = MLP().to(device)
    TOTAL_W = sum(p.numel() for p in dummy.parameters())
    K_GRID = [int(r * TOTAL_W) for r in K_RATIO_GRID]
    
    print(f"Total parameters: {TOTAL_W}")
    print(f"K_GRID: {K_GRID}")
    print(f"K_RATIO_GRID: {K_RATIO_GRID}")

    # Number of runs for statistics
    N_RUNS = 3
    results = {}

    for k in K_GRID:
        print(f"\n{'='*50}")
        print(f"Training PL0 with k={k} ({k/TOTAL_W*100:.1f}% sparsity target)")
        print(f"Running {N_RUNS} independent runs in parallel...")
        print(f"{'='*50}")
        
        # Prepare arguments for parallel execution
        run_args = []
        for run in range(N_RUNS):
            seed = 42 + run * 100 + k  # Unique seed per run and k
            run_args.append((k, device, sample_input, seed, run+1))
        
        # Execute runs in parallel using a process pool
        with Pool(processes=min(N_RUNS, cpu_count())) as pool:
            all_runs = pool.map(train_single_run, run_args)
        
        # Calculate statistics across runs
        final_accs = [run['final_acc'] for run in all_runs]
        final_sparsities = [run['final_sparsity'] for run in all_runs]
        latencies = [run['latency_ms'] for run in all_runs]
        sizes = [run['size_mb'] for run in all_runs]
        energies = [run['energy_mj'] for run in all_runs]
        
        # Store mean and std for each metric
        results[k] = {
            'acc_mean': np.mean(final_accs),
            'acc_std': np.std(final_accs),
            'sparsity_mean': np.mean(final_sparsities),
            'sparsity_std': np.std(final_sparsities),
            'latency_mean': np.mean(latencies),
            'latency_std': np.std(latencies),
            'size_mean': np.mean(sizes),
            'size_std': np.std(sizes),
            'energy_mean': np.mean(energies),
            'energy_std': np.std(energies),
            'all_runs': all_runs  # Keep individual runs for detailed analysis and temporal plots
        }
        
        print(f"\n✅ PL0 k={k} statistics across {N_RUNS} runs:")
        print(f"   Accuracy: {results[k]['acc_mean']:.4f} ± {results[k]['acc_std']:.4f}")
        print(f"   Sparsity: {results[k]['sparsity_mean']:.3f} ± {results[k]['sparsity_std']:.3f}")
        print(f"   Latency: {results[k]['latency_mean']:.2f} ± {results[k]['latency_std']:.2f} ms")
        print(f"   Size: {results[k]['size_mean']:.2f} ± {results[k]['size_std']:.2f} MB")
        print(f"   Energy: {results[k]['energy_mean']:.2f} ± {results[k]['energy_std']:.2f} mJ")

    # Save results with statistics
    save_pickle(results, "results_PL0_stats.pkl")
    print(f"\n{'='*50}")
    print("✅ Training PL0 avec statistiques terminé!")
    print("📁 Résultats sauvegardés dans 'results_PL0_stats.pkl'")
    print("📊 Prêt pour génération des graphiques avec plot_results.py")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
