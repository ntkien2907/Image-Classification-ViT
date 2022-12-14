from config import N_CLASSES
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Layer, Dense, Dropout, LayerNormalization, 
                                     MultiHeadAttention, Add, Input, Embedding, Concatenate)

MLP_DIMS = 3072
HIDDEN_DIMS = 768
N_HEADS = 12
N_PATCHES = 256
PATCH_SIZE = 32
N_CHANNELS = 3
N_LAYERS = 12


class ClassToken(Layer):
    def __init__(self):
        super().__init__()
    
    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(1, 1, input_shape[-1]), 
            dtype=tf.float32), trainable=True)
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]
        cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
        cls = tf.cast(cls, dtype=inputs.dtype)
        return cls


class MLP(Layer):
    def __init__(self):
        super().__init__()
    
    def call(self, inputs):
        x = Dense(MLP_DIMS, activation='gelu')(inputs)
        x = Dropout(0.1)(x)
        x = Dense(HIDDEN_DIMS)(x)
        x = Dropout(0.1)(x)
        return x


class TransformerEncoder(Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        skip_1 = inputs
        x = LayerNormalization()(inputs)
        x = MultiHeadAttention(num_heads=N_HEADS, key_dim=HIDDEN_DIMS)(x, x)
        x = Add()([x, skip_1])
        skip_2 = x
        x = LayerNormalization()(x)
        x = MLP()(x)
        x = Add()([x, skip_2])
        return x


def VisionTransformer():
    # Input
    input_shape = (N_PATCHES, PATCH_SIZE * PATCH_SIZE * N_CHANNELS)
    inputs = Input(input_shape)
    
    # Patch + Position Embeddings, and add class token
    patch_embed = Dense(HIDDEN_DIMS)(inputs)                # (None, 256, 768)
    position = tf.range(start=0, limit=N_PATCHES, delta=1)
    pos_embed = Embedding(input_dim=N_PATCHES, output_dim=HIDDEN_DIMS)(position)
    embed = patch_embed + pos_embed                         # (None, 256, 768)
    token = ClassToken()(embed)
    x = Concatenate(axis=1)([token, embed])                 # (None, 257, 768)

    # Stack multiple encoders
    for _ in range(N_LAYERS):
        x = TransformerEncoder()(x)

    # Classification head
    x = LayerNormalization()(x) # (None, 257, 768)
    x = x[:, 0, :]              # (None, 768)
    outputs = Dense(N_CLASSES, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model
    

# model = VisionTransformer()
# model.summary()