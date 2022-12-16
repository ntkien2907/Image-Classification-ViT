DATASET_DIR = 'data/flowers/'
MODEL_PATH  = 'output/ViT_model.h5'
CSV_PATH    = 'output/ViT_log.csv'
ACC_LOSS    = 'output/ViT_accuracy-and-loss.png'

PARAMS = {
    'IMAGE_SIZE': 200, 
    'N_CHANNELS': 3, 
    'PATCH_SIZE': 25, 
    'BATCH_SIZE': 10, 
    'LR': 1e-3, 
    'N_EPOCHS': 20, 
    'CLASS_NAMES': ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'], 

    'N_LAYERS': 2, 
    'HIDDEN_DIMS': 512, 
    'MLP_DIMS': 2048, 
    'N_HEADS': 8, 
}
RANDOM_STATE = 42

# ===================================================================================================================== #

PARAMS['N_CLASSES'] = len(PARAMS['CLASS_NAMES'])
PARAMS['N_PATCHES'] = PARAMS['IMAGE_SIZE']**2 // PARAMS['PATCH_SIZE']**2
PARAMS['FLAT_PATHCHES_SHAPE'] = (PARAMS['N_PATCHES'], PARAMS['PATCH_SIZE'] * PARAMS['PATCH_SIZE'] * PARAMS['N_CHANNELS'])