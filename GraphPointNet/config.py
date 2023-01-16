DEVICE = "cuda:0" # cuda:0
DEBUG = False

# -----------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------
DATASET = dict(
    NUM_CONNECTIONS = 10,
    NUMBER_OF_POINTS = 2500
)

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
DATALOADER = dict(
    BATCH_SIZE = 4,
    NUM_WORKERS = 4
)

MODEL = dict(
)

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
SOLVER = dict(
    LR = 0.001,
    EPOCHS = 300,
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
    LOAD_CHECKPOINT = False,
    LOAD_CHECKPOINT_PATH = "checkpoints/GPN/GPN.pth",
    WANDB_LOG = False,
    WANDB_PROJECT = "GPN",
)

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
TEST = dict(
    LOAD_CHECKPOINT_PATH = "checkpoints/GPN/GPN_77e.pth",
    VISUALIZE = False,
)