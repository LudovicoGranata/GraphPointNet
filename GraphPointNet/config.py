DEVICE = "cuda:0" # cuda:0
DEBUG = dict(
    ENABLE = False,
    NUM_SAMPLES = 20,
    )

# -----------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------
DATASET = dict(
    NUM_CONNECTIONS = 3,
    NUMBER_OF_POINTS = 2048,
    NORMAL_CHANNEL = True
    )

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
DATALOADER = dict(
    BATCH_SIZE = 4,
    NUM_WORKERS = 4,
    CACHE = True,
)

MODEL = dict(
    NAME = "GPN" # GPN | PNPP
)

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
SOLVER = dict(
    LR = 0.001,
    EPOCHS = 300,
    MIN_LR = 1e-5,
    LR_DECAY = 0.5,
    LR_STEP_SIZE = 20
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