from utils import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from ViT import VisionTransformer

# Dataset
X_train, X_valid, _ = load_data(DATASET_DIR)
train_ds = tf_dataset(X_train, PARAMS['BATCH_SIZE'])
valid_ds = tf_dataset(X_valid, PARAMS['BATCH_SIZE'])

# Model
model = VisionTransformer(PARAMS)
adam = Adam(learning_rate=PARAMS['LR'], clipvalue=1.0)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])

callbacks = [
    ModelCheckpoint(MODEL_PATH, monitor='val_loss', verbose=1, save_best_only=True), 
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-10, verbose=1), 
    CSVLogger(CSV_PATH), 
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=False), 
]

history = model.fit(train_ds, epochs=PARAMS['N_EPOCHS'], validation_data=valid_ds, callbacks=callbacks)
save_figures(history, ACC_LOSS)