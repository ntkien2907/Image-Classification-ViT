from utils import *
from ViT import VisionTransformer

# Dataset
X_train, X_valid, X_test = load_data(DATASET_DIR)
train_ds = tf_dataset(X_train, PARAMS['BATCH_SIZE'])
valid_ds = tf_dataset(X_valid, PARAMS['BATCH_SIZE'])

# # Model
# model = VisionTransformer()