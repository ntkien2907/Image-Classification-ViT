DATASET_DIR = 'data/flowers/'
OUTPUT_DIR  = 'output'
MODEL_NAME  = 'FineTunedViT'             # ViT or FineTunedViT

PARAMS = {
    'IMAGE_SIZE': 224, 
    'N_CHANNELS': 3, 
    'PATCH_SIZE': 28, 

    'LR': 1e-4, 
    'WD': 1e-4, 
    'DROP_RATE': 0.1, 
    'NORM_EPS': 1e-12, 
    
    'BATCH_SIZE': 32, 
    'N_EPOCHS': 30, 
    'CLASS_NAMES': ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'], 

    'N_LAYERS': 10, 
    'HIDDEN_DIMS': 128, 
    'MLP_DIMS': 512, 
    'N_HEADS': 10, 
}
RANDOM_STATE = 42

# ===================================================================================================================== #

PARAMS['IMAGE_SHAPE'] = (PARAMS['IMAGE_SIZE'], PARAMS['IMAGE_SIZE'], PARAMS['N_CHANNELS'])

PARAMS['N_CLASSES'] = len(PARAMS['CLASS_NAMES'])
PARAMS['N_PATCHES'] = PARAMS['IMAGE_SIZE']**2 // PARAMS['PATCH_SIZE']**2
PARAMS['FLAT_PATHCHES_SHAPE'] = (PARAMS['N_PATCHES'], PARAMS['PATCH_SIZE'] * PARAMS['PATCH_SIZE'] * PARAMS['N_CHANNELS'])

MODEL_PATH = f'{OUTPUT_DIR}/{MODEL_NAME}_model.h5'
CSV_PATH   = f'{OUTPUT_DIR}/{MODEL_NAME}_log.csv'
ACC_LOSS   = f'{OUTPUT_DIR}/{MODEL_NAME}_accuracy-and-loss.png'
CLS_REPORT = f'{OUTPUT_DIR}/{MODEL_NAME}_confusion-matrix.csv'