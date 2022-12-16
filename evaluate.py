import pandas as pd
from utils import *
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from ViT import VisionTransformer

# Load test set
_, _, X_test = load_data(DATASET_DIR)
test_ds = tf_dataset(X_test, PARAMS['BATCH_SIZE'])
y_true = [np.argmax(y_onehot) for _, labels in list(test_ds) for y_onehot in labels]

# Load model and its weight
model = VisionTransformer(PARAMS)
model.load_weights(MODEL_PATH)
adam = Adam(PARAMS['LR'], beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])

# Predict
y_pred = model.predict(test_ds)
y_pred = [np.argmax(y_onehot) for y_onehot in y_pred]

# Confusion matrix
cls_report = classification_report(y_true, y_pred, target_names=PARAMS['CLASS_NAMES'], output_dict=True)
cls_report.update({'accuracy': {'precision': None, 'recall': None, 'f1-score': cls_report['accuracy'], 'support': cls_report['macro avg']['support']}})
df = pd.DataFrame(cls_report).transpose()
df.to_csv(CLS_REPORT)