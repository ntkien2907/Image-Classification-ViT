from utils import *
from tensorflow.keras.optimizers import Adam
from ViT import VisionTransformer

# Load test set
_, _, X_test = load_data(DATASET_DIR)
test_ds = tf_dataset(X_test, PARAMS['BATCH_SIZE'])

# Load model and its weight
model = VisionTransformer(PARAMS)
model.load_weights(MODEL_PATH)
adam = Adam(PARAMS['LR'], beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])

# Evaluate
model.evaluate(test_ds)