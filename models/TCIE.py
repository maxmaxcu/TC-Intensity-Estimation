from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D, \
    MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Input
from tensorflow.keras.regularizers import l1_l2, L1
from tensorflow.keras.initializers import RandomNormal

from tensorflow import keras

def cus_loss(y_true, y_pred):
    loss1 = keras.losses.categorical_crossentropy(y_true[0], y_pred[0])
    loss2 = keras.losses.mean_squared_error(y_true[0], y_pred[0]) * 0.005
    return loss1 + loss2

def TCIE(W_l1RE, shape, dropout_net=0.5, model_type=''):
    inputs = Input(shape=shape)
    x = Conv2D(64, (10, 10), strides=1, padding='valid',
               kernel_regularizer=L1(W_l1RE))(inputs)

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

    # Let's train the model using RMSprop
    # model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    if model_type in ['cls1', 'cls2']:
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
    else:
        model.compile(loss=cus_loss,
                      optimizer=opt, metrics={'dense_2': 'accuracy', 'dense_3': 'mse'})

    model.summary()
    return model