import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.losses import CategoricalCrossentropy
from patchify import patchify
from utils import classifier
from config import *

st.set_page_config(page_title='Flowers Classification ViT', 
                   page_icon='https://cdn-icons-png.flaticon.com/512/3200/3200079.png', 
                   layout='centered')

st.header('Image Classification using Vision Transformer')

# ===================================================================================================================== #

model = classifier(PARAMS)
model.load_weights(MODEL_PATH)
optimizer = AdamW(learning_rate=PARAMS['LR'], weight_decay=PARAMS['WD'])
loss = CategoricalCrossentropy(label_smoothing=0.2)
model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])

# ===================================================================================================================== #

def predict(img_arr, model):
    img_arr = cv2.resize(img_arr, (PARAMS['IMAGE_SIZE'], PARAMS['IMAGE_SIZE']), cv2.INTER_CUBIC)
    img_arr = img_arr / 255.0

    if MODEL_NAME == 'ViT':
        patch_shape = (PARAMS['PATCH_SIZE'], PARAMS['PATCH_SIZE'], PARAMS['N_CHANNELS'])
        patches = patchify(img_arr, patch_shape, PARAMS['PATCH_SIZE'])
        patches = np.reshape(patches, PARAMS['FLAT_PATHCHES_SHAPE'])
    
    elif MODEL_NAME == 'FineTunedViT':
        patches = img_arr
    
    patches = patches.astype(np.float32)
    patches = np.expand_dims(patches, axis=0)

    y_pred = model(patches)
    class_idx = np.argmax(y_pred)
    class_prob = float(y_pred[0][class_idx])
    class_name = PARAMS['CLASS_NAMES'][class_idx]

    print(f'[INFO] Probability: {class_prob:4f}, Class: {class_name}')
    if class_prob < 0.65: class_name = 'unknown'
    
    return class_name

# ===================================================================================================================== #

uploaded_file = st.file_uploader(label='')

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    img_arr = np.asarray(Image.open(io.BytesIO(bytes_data)).convert('RGB'))
    class_name = predict(img_arr, model)
    img_arr = cv2.putText(img_arr, class_name, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2, cv2.LINE_AA)
    st.image(img_arr)