# src/config.py
DATA_ROOT = "./data"       # <- on veut un seul répertoire "data"
BATCH_SIZE = 128
EPOCHS = 20
LR = 0.01
MOMENTUM = 0.9

# P-L1
LAMBDA_GRID = [1e-5, 1e-4, 1e-3]
# P-L0
K_RATIO_GRID = [0.20, 0.10, 0.05]

# config.py
DEVICE = 'cuda'  # or 'cpu'
BENCHMARK = {
    'n_warmup': 10,
    'n_runs': 100,
    'cpu_power_w': 10.0
}
