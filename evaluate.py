import pandas as pd
from utils import *
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.metrics import classification_report

# Load test set
_, _, X_test = load_data(DATASET_DIR)
test_ds = tf_dataset(X_test, PARAMS['BATCH_SIZE'])
y_true = [np.argmax(y_onehot) for _, labels in list(test_ds) for y_onehot in labels]

# Load model and its weight
model = classifier(PARAMS)
model.load_weights(MODEL_PATH)
optimizer = AdamW(learning_rate=PARAMS['LR'], weight_decay=PARAMS['WD'])
loss = CategoricalCrossentropy(label_smoothing=0.2)
model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])

# Predict
y_pred = model.predict(test_ds)
y_pred = [np.argmax(y_onehot) for y_onehot in y_pred]

# Confusion matrix
cls_report = classification_report(y_true, y_pred, target_names=PARAMS['CLASS_NAMES'], output_dict=True)
cls_report.update({'accuracy': {'precision': None, 'recall': None, 'f1-score': cls_report['accuracy'], 'support': cls_report['macro avg']['support']}})
df = pd.DataFrame(cls_report).transpose()
df.to_csv(CLS_REPORT, float_format='%.4f')