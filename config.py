DATASET_DIR = 'flowers/'
MODEL_PATH  = 'output/model.h5'
CSV_PATH    = 'output/log.csv'

IMAGE_SIZE = 200
N_CHANNELS = 3
PATCH_SIZE = 25

BATCH_SIZE = 16
LR = 1e-4
N_EPOCHS = 50

RANDOM_STATE = 2

# ======================================================================== #

N_PATHCHES = IMAGE_SIZE**2 // PATCH_SIZE**2
FLAT_PATHCHES_SHAPE = (N_PATHCHES, PATCH_SIZE * PATCH_SIZE * N_CHANNELS)

CLASS_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
N_CLASSES = len(CLASS_NAMES)