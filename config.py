DATASET_DIR = 'data/flowers/'
OUTPUT_DIR  = 'output'
MODEL_PATH  = f'{OUTPUT_DIR}/ViT_model.h5'
CSV_PATH    = f'{OUTPUT_DIR}/ViT_log.csv'
ACC_LOSS    = f'{OUTPUT_DIR}/ViT_accuracy-and-loss.png'
CLS_REPORT  = f'{OUTPUT_DIR}/ViT_confusion-matrix.csv'

PARAMS = {
    'IMAGE_SIZE': 200, 
    'N_CHANNELS': 3, 
    'PATCH_SIZE': 25, 
    'BATCH_SIZE': 32, 
    'LR': 1e-4, 
    'N_EPOCHS': 30, 
    'CLASS_NAMES': ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'], 

    'N_LAYERS': 4, 
    'HIDDEN_DIMS': 128, 
    'MLP_DIMS': 512, 
    'N_HEADS': 8, 
}
RANDOM_STATE = 42

# ===================================================================================================================== #

PARAMS['N_CLASSES'] = len(PARAMS['CLASS_NAMES'])
PARAMS['N_PATCHES'] = PARAMS['IMAGE_SIZE']**2 // PARAMS['PATCH_SIZE']**2
PARAMS['FLAT_PATHCHES_SHAPE'] = (PARAMS['N_PATCHES'], PARAMS['PATCH_SIZE'] * PARAMS['PATCH_SIZE'] * PARAMS['N_CHANNELS'])