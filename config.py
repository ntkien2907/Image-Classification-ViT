DATASET_DIR = 'flowers/'

IMAGE_SIZE = 200
N_CHANNELS = 3
PATCH_SIZE = 25
N_PATHCHES = IMAGE_SIZE**2 // PATCH_SIZE**2

FLAT_PATHCHES_SHAPE = (N_PATHCHES, PATCH_SIZE * PATCH_SIZE * N_CHANNELS)