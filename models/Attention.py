from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D, Multiply, Add, Permute, Lambda, \
    MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Input, GlobalMaxPooling2D, Reshape, Concatenate
from tensorflow.keras.regularizers import l1_l2, L1
from tensorflow.keras.initializers import RandomNormal

from tensorflow import keras
import tensorflow.keras.backend as K

def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature


def channel_attention(input_feature, ratio=8):
    channel_axis = -1
    channel = input_feature._keras_shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return Multiply()([input_feature, cbam_feature])


def spatial_attention(input_feature):
    kernel_size = 7

    cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature._keras_shape[-1] == 1

    return Multiply()([input_feature, cbam_feature])

def AttentionNetwork(W_l1RE, shape, dropout_net=0.5, model_type=''):
    inputs = Input(shape=shape)
    x = cbam_block(inputs)
    x = Conv2D(64, (10, 10), strides=1, padding='valid',
               kernel_regularizer=L1(W_l1RE))(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)

    x = Conv2D(256, (5, 5), strides=1, dilation_rate=(2, 2), kernel_regularizer=L1(W_l1RE))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)

    x = Conv2D(288, (3, 3), strides=1, padding='same', dilation_rate=(2, 2), kernel_regularizer=L1(W_l1RE))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x1 = MaxPooling2D(pool_size=(2, 2), strides=1)(x)

    x = Conv2D(272, (3, 3), strides=1, padding='same', dilation_rate=(2, 2), kernel_regularizer=L1(W_l1RE))(x1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), strides=1, padding='same', kernel_regularizer=L1(W_l1RE))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(2048, kernel_regularizer=L1(W_l1RE))(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_net)(x)
    x = Dense(1024, kernel_regularizer=L1(W_l1RE))(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_net)(x)
    if 'cls1' == model_type:
        x = Dense(7, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                  kernel_regularizer=l1_l2(W_l1RE, l2=0), activation='softmax')(x)
        model = Model([inputs], [x])
    elif 'cls2' == model_type:
        x = Dense(9, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                  kernel_regularizer=l1_l2(W_l1RE, l2=0), activation='softmax')(x)
        model = Model([inputs], [x])
    else:
        if 'cls1' in model_type:
            x1 = Dense(7, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                      kernel_regularizer=l1_l2(W_l1RE, l2=0), activation='softmax')(x)
            x2 = Dense(1, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                      kernel_regularizer=l1_l2(W_l1RE, l2=0), activation='linear')(x)
            model = Model([inputs], [x1, x2])
        elif 'cls2' in model_type:
            x1 = Dense(9, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                      kernel_regularizer=l1_l2(W_l1RE, l2=0), activation='softmax')(x)
            x2 = Dense(1, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                      kernel_regularizer=l1_l2(W_l1RE, l2=0), activation='linear')(x)
            model = Model([inputs], [x1, x2])

    opt = keras.optimizers.SGD(lr=0.0005, decay=1e-6, momentum=0.9)

    if model_type in ['cls1', 'cls2']:
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
    else:
        model.compile(loss=cus_loss,
                      optimizer=opt, metrics={'dense_2': 'accuracy', 'dense_3': 'mse'})

    model.summary()
    return model


def cus_loss(y_true, y_pred):
    loss1 = keras.losses.categorical_crossentropy(y_true[0], y_pred[0])
    loss2 = keras.losses.mean_squared_error(y_true[0], y_pred[0]) * 0.005
    return loss1 + loss2

