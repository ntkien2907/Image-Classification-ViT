## Image Classification using Vision Transformer

### Introduction
The Vision Transformer, or **ViT**, is a model for image classification that employs a Transformer-like architecture over patches of the image. An image is split into fixed-size patches, each of them are then linearly embedded, position embeddings are added, and the resulting sequence of vectors is fed to a standard Transformer encoder. In order to perform classification, the standard approach of adding an extra learnable “classification token” to the sequence is used.

### Vision Transformer

#### Architecture
<p align='middle'><img src='./assets/ViT-architecture.png' width=45% /></p>

#### Mechanism
<p align='middle'><img src='./assets/ViT-mechanism.gif' width=45% /></p>

### Dataset
* The dataset contains 4242 images of flowers, and is divided into 5 classes: *chamomile*, *tulip*, *rose*, *sunflower*, *dandelion*.
* For each class there are about 800 photos. Photos are not high resolution, about 320x240 pixels. Photos are not reduced to a single size, they have different proportions.
* Download from [Kaggle](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition).

### How to use
* Step 1: Automatically install all dependencies from `requirements.txt`.
    ```
    pip install -r requirements.txt
    ```
* Step 2: Configure hyper-parameters in `config.py` and train model.
    ```
    python train.py
    ```
* Step 3: Evaluate model.
    ```
    python evaluate.py
    ```

### References
[Alexey Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale". In ICLR, 2021.](https://arxiv.org/abs/2010.11929)