DATASET_DIR = 'flowers/'
MODEL_PATH  = 'output/model.h5'
CSV_PATH    = 'output/log.csv'

PARAMS = {
    'IMAGE_SIZE': 200, 
    'N_CHANNELS': 3, 
    'PATCH_SIZE': 25, 
    'BATCH_SIZE': 16, 
    'LR': 1e-4, 
    'N_EPOCHS': 50, 
    'CLASS_NAMES': ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
}

# ======================================================================================================================= #

PARAMS['N_CLASSES']  = len(PARAMS['CLASS_NAMES'])
PARAMS['N_PATHCHES'] = PARAMS['IMAGE_SIZE']**2 // PARAMS['PATCH_SIZE']**2
PARAMS['FLAT_PATHCHES_SHAPE'] = (PARAMS['N_PATHCHES'], PARAMS['PATCH_SIZE'] * PARAMS['PATCH_SIZE'] * PARAMS['N_CHANNELS'])