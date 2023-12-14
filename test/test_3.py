import tensorflow as tf
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.utils import plot_model
from SPP3 import spatial_pyramid_pooling

input_shape = (128, 128, 3)
left_input = Input(shape=input_shape)
right_input = Input(shape=input_shape)

def model_1(inputs):
# Model_1
    x1 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(inputs)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = spatial_pyramid_pooling(x1)
    x1 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.MaxPooling2D((2, 2))(x1)
    return x1

encoded_l = model_1(left_input)
encoded_r = model_1(right_input)

merger = concatenate([encoded_l, encoded_r], axis=-1)

siamese_network = tf.keras.models.Model(inputs=[left_input, right_input], outputs=merger)
siamese_network.summary()
# plot_model(siamese_network, to_file='siamese_network.png', show_shapes=True)

# Model_2
model_2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu', input_shape=siamese_network.output_shape[1:]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(9, activation='softmax'),
])
# model_2.summary()

final = model_2(siamese_network.output)

final_model = tf.keras.models.Model(inputs=siamese_network.inputs, outputs=final)

# final_model.summary()
# plot_model(final_model, to_file='final_model.png', show_shapes=True)