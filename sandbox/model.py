import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical


OUTPUT_CHANNELS = 1

def frontend_feat(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = keras.Sequential()
    
    if filters == 'M':
        result.add(
            keras.layers.MaxPooling2D())
    else:
        result.add(
          keras.layers.Conv2D(filters, size, padding='same',
                                 kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(keras.layers.BatchNormalization())

        result.add(keras.layers.ReLU())

    return result

def backend_feat(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = keras.Sequential()
    result.add(
    keras.layers.Conv2DTranspose(filters, size, padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    result.add(keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(keras.layers.Dropout(0.5))

    result.add(keras.layers.ReLU())

    return result

def create_model():
    frontend_stack = [
        frontend_feat(64, 3, apply_batchnorm= False),
        frontend_feat(64, 3),
        frontend_feat('M', 2),
        frontend_feat(128, 3),
        frontend_feat(128, 3),
        frontend_feat('M', 2),
        frontend_feat(256, 3),
        frontend_feat(256, 3),
        frontend_feat(256, 3),
        frontend_feat('M', 2),
        frontend_feat(512, 3),
        frontend_feat(512, 3),
        frontend_feat(512, 3),
    ]    
    backend_stack = [
        backend_feat(512, 3),
        backend_feat(512, 3),
        backend_feat(512, 3),
        backend_feat(256, 3),
        backend_feat(128, 3),
        backend_feat(64, 3),
    ]
    
    initializer = tf.random_normal_initializer(0., 0.02)
    last_layer = keras.layers.Dense(OUTPUT_CHANNELS)
    concat = keras.layers.Concatenate()
    inputs = keras.layers.Input(shape = (None,None,3))
    x = inputs
    
    skips = []
    for front in frontend_stack:
        x = front(x)
        skips.append(x)
        
    skips = reversed(skips[:-1])
    
    for back, skip in zip(backend_stack, skips):
        x = back(x)
        x = concat([x, skip])
        
    x = last_layer(x)
    
    return keras.Model(inputs = inputs, outputs = x)