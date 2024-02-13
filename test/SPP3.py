from tensorflow.keras.layers import concatenate, Conv2D, AveragePooling2D, UpSampling2D

def spatial_pyramid_pooling(x):
    # SPP with fixed-size average pooling blocks: 64x64, 32x32, 16x16, and 8x8
    pool_1 = AveragePooling2D(pool_size=(64, 64))(x)
    pool_1 = Conv2D(128, (1, 1), activation='relu', padding='same')(pool_1)
    pool_1 = UpSampling2D(size=(64, 64))(pool_1)
    
    pool_2 = AveragePooling2D(pool_size=(32, 32))(x)
    pool_2 = Conv2D(128, (1, 1), activation='relu', padding='same')(pool_2)
    pool_2 = UpSampling2D(size=(32, 32))(pool_2)
    
    pool_3 = AveragePooling2D(pool_size=(16, 16))(x)
    pool_3 = Conv2D(128, (1, 1), activation='relu', padding='same')(pool_3)
    pool_3 = UpSampling2D(size=(16, 16))(pool_3)
    
    pool_4 = AveragePooling2D(pool_size=(8, 8))(x)
    pool_4 = Conv2D(128, (1, 1), activation='relu', padding='same')(pool_4)
    pool_4 = UpSampling2D(size=(8, 8))(pool_4)
    
    # Concatenate the pooled features
    spp = concatenate([pool_1, pool_2, pool_3, pool_4], axis=-1)

    return spp