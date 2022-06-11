from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, \
    MaxPooling2D, BatchNormalization, Input, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l1_l2, L1
from tensorflow.keras.initializers import RandomNormal

from tensorflow import keras


def cus_loss(y_true, y_pred):
    loss1 = keras.losses.categorical_crossentropy(y_true[0], y_pred[0])
    loss2 = keras.losses.mean_squared_error(y_true[0], y_pred[0]) * 0.005
    return loss1 + loss2

def vgg2(W_l1RE, W_l2RE, shape, dropout_net, model_type='reg'):
    inputs = Input(shape=shape)
    x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                     input_shape=shape, kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(inputs)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01)
                     , kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01)
                     , kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (3, 3), padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01)
                     , kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01)
                     , kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, (3, 3), padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01)
                     , kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(512, (3, 3), padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01)
                     , kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(512, (3, 3), padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01)
                     , kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01)
                     , kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(512, (3, 3), padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01)
                     , kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    x = Dense(512, kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Dropout(dropout_net)(x)
    x = Dense(64, kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(x)
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
    elif model_type == 'reg':
        x = Dense(1, kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(x)
        x = Activation('linear')(x)
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

    opt = keras.optimizers.SGD(lr=0.0002, decay=1e-6, momentum=0.9)


    if model_type in ['cls1', 'cls2']:
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
    elif model_type == 'reg':
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
    else:
        model.compile(loss=cus_loss,
                      optimizer=opt, metrics={'dense_2': 'accuracy', 'dense_3': 'mse'})
    model.summary()
    return model


