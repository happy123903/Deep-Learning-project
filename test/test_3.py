import tensorflow as tf
from tensorflow.keras.layers import Input, concatenate
import numpy as np
import cv2
from matplotlib import pyplot as plt
# from tensorflow.keras.utils import plot_model
from SPP3 import spatial_pyramid_pooling

# 讀取訓練圖片B
train_images = np.load('C:\\Users\\88696\\Desktop\\專題研究\\main\\train_images.npy', allow_pickle=True)

# 讀取訓練圖片A
train_images_A = cv2.imread('../A.jpg')
train_images_A = cv2.resize(train_images_A, (128, 128))
train_images_A = train_images_A / 255
train_images_A = np.expand_dims(train_images_A, axis=0)
train_images_A_fixed = np.tile(train_images_A, (len(train_images), 1, 1, 1)) #複製圖片A的數量使之與B相同

# 讀取訓練標籤
train_labels = np.load('C:\\Users\\88696\\Desktop\\專題研究\\main\\labels.npy', allow_pickle=True)

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

# Model_2
model_2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu', input_shape=siamese_network.output_shape[1:]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((4,4)),
    tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(9, activation='softmax'),
])

final = model_2(siamese_network.output)

final_model = tf.keras.models.Model(inputs=siamese_network.inputs, outputs=final)
# final_model.summary()

# 選擇優化器和損失函數
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
final_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

history = final_model.fit([train_images_A_fixed, train_images], train_labels, batch_size=10, epochs=10, validation_split=0.2)
print(history.history.keys())

'''
# accuracy plt
plt.rcParams['font.sans-serif']=['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus']=False

plt.Figure(figsize=(8,6))
plt.plot(history.history['accuracy'], 'r', label='訓練準確度')
plt.plot(history.history['val_accuracy'], 'g', label='驗證準確度')
plt.legend()
plt.show()
# loss plt
plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], 'r', label='訓練損失')
plt.plot(history.history['val_loss'], 'g', label='驗證損失')
plt.legend()
plt.show()
'''