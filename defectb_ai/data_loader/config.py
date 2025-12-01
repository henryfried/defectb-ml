# Training hyperparameters
INPUT_SIZE = 600
TARGET_SIZE = 2
NUM_CLASSES = 2
LEARNING_RATE = 0.0241
BATCH_SIZE = 369
NUM_EPOCHS = 200
DROPOUT = 0.0
ALPHA = 0.0

OUTPUT_DIMS = [109]


# Dataset
# DATA_DIR = f"../../../../hBN_supercell/data_sets"
DATA_DIR = "/Users/henry.fried/hpc_mount/thirdyear/ml_hBN/data_sets"

# TRAIN_SET = "sigma_u_only"
# TRAIN_SET = "laplace/nn_1/u300_3-5_sigma100_0.05-0.25_"
TRAIN_SET = "sigma_u_only_-4_4_500_0005_025_100"
PARAMETER = 'both'
NUM_WORKERS = 0
TARGET_MIN_MAX = [[-4, 4], [0.005, 0.25], [-2.5, -3.5], [-1, -0], [-1, -0]]
# Compute related
ACCELERATOR = "cpu"
DEVICES = 1
PRECISION = 32
#

TRAINED_MODEL_DIR = "lightning_logs/very_good/checkpoints/epoch=199-step=19000.ckpt"
#conv
#TRAINED_MODEL_DIR = "tb_logs/conv/laplace/nn_1/u200_3-5_sigma100_0.05-0.25_/version_0/checkpoints/epoch=99-step=128000.ckpt"

PRED_DATA_DIR = '../../../ldos_data/c_n/dft.dat'