import torch, time, pickle, json
import torch.nn as nn

def accuracy(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
    return correct / len(loader.dataset)

def sparsity(model, tol=1e-8):
    total = zero = 0
    for p in model.parameters():
        tensor = p.data
        total += tensor.numel()
        zero  += (tensor.abs() < tol).sum().item()
    return zero / total

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def epoch_time(start_time):
    return time.time() - start_time

# ── Green AI benchmarking helpers ─────────────────────────────────────────────

CPU_POWER_W = 10.0  # assumed CPU power draw in watts for energy estimate

def benchmark_latency(model: torch.nn.Module,
                      input_tensor: torch.Tensor,
                      device: torch.device,
                      n_warmup: int = 10,
                      n_runs: int = 100) -> float:
    """Return average inference time in milliseconds over n_runs."""
    model.to(device).eval()
    x = input_tensor.to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = model(x)
        end = time.perf_counter()
    return (end - start) / n_runs * 1000

def model_size_mb(model: torch.nn.Module, tol: float = 1e-8) -> float:
    """
    Return the size in MB counting only non-zero parameters (float32),
    so that pruning se traduit bien par une réduction de taille.
    """
    nonzero = sum((p.data.abs() > tol).sum().item() for p in model.parameters())
    return nonzero * 4 / (1024 ** 2)


def estimate_energy_mj(latency_ms: float,
                       cpu_power_w: float = CPU_POWER_W) -> float:
    """Estimate energy (mJ) = Power(W) × time(s) × 1000."""
    return cpu_power_w * (latency_ms / 1000) * 1000


if __name__ == "__main__":
    print("✔️ utils.py loaded successfully.")
