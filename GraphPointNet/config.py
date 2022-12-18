DEVICE = "cuda:0" # cuda:0
DEBUG = False

# -----------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------
DATASET = dict(
)

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
DATALOADER = dict(
    BATCH_SIZE = 4,
    NUM_WORKERS = 3
)

MODEL = dict(
)

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
SOLVER = dict(
    LR = 0.001,
    EPOCHS = 100,
    SCHEDULER = True,
    SCHEDULER_NAME = "ExponentialLR",
    GAMMA = 0.98,
)

# ---------------------------------------------------------------------------- #
# Specific train options
# ---------------------------------------------------------------------------- #
TRAIN = dict(
    SAVE_CHECKPOINT = True,
    SAVE_CHECKPOINT_PATH = "checkpoints/GPN/GPN.pth",
    LOAD_CHECKPOINT = True,
    LOAD_CHECKPOINT_PATH = "checkpoints/GPN/GPN.pth",
    WANDB_LOG = False,
    WANDB_PROJECT = "GPN",
)

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
TEST = dict(
    LOAD_CHECKPOINT_PATH = "checkpoints/GPN/GPN.pth",
)