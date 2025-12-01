#data related
SEP_DOS_SCALE = True
INCLUDE_SUM = True
#NUM_DATA = 200
LDOS_NUM = 4
NEDOS = 400
#input size depends on number of PDOS and number of steps for (NEDOS)
INPUT_SIZE = (LDOS_NUM+1)*NEDOS
# without sum included
# INPUT_SIZE = (LDOS_NUM)*NEDOS
TARGET_SIZE = 5

#trainer related
DECAY_RATE = 0.99
LEARNING_RATE =1e-3
BATCH_SIZE = 4
NUM_EPOCHS = 1
DROPOUT = 0.0
ALPHA = 0.0
#attention variables
NUM_HEADS = 1
HEAD_DIM = 2

#network architecture
# CONV_LAYER = [[4, 32, 8, 0]]
# OUTPUT_DIMS = [128, 256, 512, 256, 64]
CONV_LAYER = [[2, 4, 8, 0]]
OUTPUT_DIMS = [128]

#data related

# Compute related
ACCELERATOR = "cpu"
DEVICES = 1  
PRECISION = 32
NUM_WORKERS = 1 #DEVICES
#-----------------------------------------------------------------------------
#				        C_N	
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#				       1 gauss	
#-----------------------------------------------------------------------------
from pathlib import Path

TRAIN_SET_NAME = 'c_dimer_nedos_400_tb'
BASE_DIR = Path(__file__).resolve().parent.parent  # defectb_ai/examples
DATA_DIR = BASE_DIR / "data_sets"

TRAIN_SET = str(DATA_DIR / TRAIN_SET_NAME)

VERSION_NUM = 0
MODEL = 'epoch=0-step=200.ckpt'
TRAINED_MODEL_DIR = f'tb_logs/{TRAIN_SET_NAME}/version_{VERSION_NUM}/checkpoints/{MODEL}'

PRED_DATA_DIR = str(DATA_DIR / "c_dimer_nedos_400_dft.dat")
# PRED_DATA_DIR = f'{START_DIR}/hbn_data_sets/dft/c_n/c_n_ldos_num_{LDOS_NUM}_nedos_{NEDOS}.dat'
