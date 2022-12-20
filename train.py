from utils import *
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping

# Make output directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load dataset
X_train, X_valid, _ = load_data(DATASET_DIR)
train_ds = tf_dataset(X_train, PARAMS['BATCH_SIZE'])
valid_ds = tf_dataset(X_valid, PARAMS['BATCH_SIZE'])

# Model
model = classifier(MODEL_NAME, PARAMS)
optimizer = AdamW(learning_rate=PARAMS['LR'], weight_decay=PARAMS['WD'])
loss = CategoricalCrossentropy(label_smoothing=0.2)
model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])

callbacks = [
    ModelCheckpoint(MODEL_PATH, monitor='val_loss', verbose=1, save_best_only=True), 
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-10, verbose=1), 
    CSVLogger(CSV_PATH), 
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=False), 
]

# Train model
history = model.fit(train_ds, epochs=PARAMS['N_EPOCHS'], validation_data=valid_ds, callbacks=callbacks)
save_figures(history, ACC_LOSS)