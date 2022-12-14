from utils import *

# Dataset
X_train, X_val, X_test = load_data(DATASET_DIR)
train_ds = tf_dataset(X_train, BATCH_SIZE)
val_ds = tf_dataset(X_val, BATCH_SIZE)

# Model
