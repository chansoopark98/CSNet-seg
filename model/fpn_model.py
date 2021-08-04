from tensorflow.keras.layers import (
    MaxPooling2D, SeparableConv2D, UpSampling2D, Activation, Add, BatchNormalization, Conv2D, Dropout, Concatenate)
from functools import reduce

MOMENTUM = 0.99
EPSILON = 0.0001

def fpn_model(features, fpn_times=3, activation='swish'):
    c3, c4, c5 = features

    c6 = MaxPooling2D(pool_size=3, strides=2, padding='same')(c5)
    c7 = MaxPooling2D(pool_size=3, strides=2, padding='same')(c6)

    features = [c3, c4, c5, c6, c7]

    for i in range(fpn_times):
        features = _build_fpn(features, num_channels=128, activation=activation)

    x1 = features[0]
    x2 = features[1]
    x3 = features[2]
    x4 = features[3]
    x5 = features[4]

    x1 = _convBlock(x=x1, num_channels=128, kernel_size=3, strides=2, name='x1_feature_pool')
    x1 = Activation(activation)(x1)
    x1 = MaxPooling2D(pool_size=3, strides=2, padding='same')(x1) # 128 to 32

    x2 = _convBlock(x=x2, num_channels=128, kernel_size=3, strides=2, name='x2_feature_pool')
    x2 = Activation(activation)(x2)

    x4 = UpSampling2D(size=(2, 2), interpolation='bilinear')(x4)
    x4 = _convBlock(x=x4, num_channels=128, kernel_size=3, strides=1, name='x4_feature_pool')
    x4 = Activation(activation)(x4)

    x5 = UpSampling2D(size=(4, 4), interpolation='bilinear')(x5)
    x5 = _convBlock(x=x5, num_channels=128, kernel_size=3, strides=1, name='x5_feature_pool')
    x5 = Activation(activation)(x5)

    x = Concatenate()([x1, x2, x3, x4, x5])
    x = _convBlock(x=x, num_channels=512, kernel_size=3, strides=1, name='refining_process')
    x = Activation(activation)(x)
    x = Dropout(rate=0.5)(x)

    x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    x = _convBlock(x=x, num_channels=256, kernel_size=3, strides=1, name='up4x_sep_conv')
    x = Activation(activation)(x)
    x = Dropout(rate=0.3)(x)
    x = _convBlock(x=x, num_channels=256, kernel_size=3, strides=1, name='conv_block')
    x = Activation(activation)(x)

    return x

def _convBlock(x, num_channels, kernel_size, strides, name, dilation_rate=1):
    x = Conv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
               use_bias=True)(x)
    x = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(x)
    return x


def _separableConvBlock(num_channels, kernel_size, strides, dilation_rate=1):
    # f1 = SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
    #                      use_bias=True, dilation_rate=dilation_rate)
    f1 = Conv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                         use_bias=True, dilation_rate=dilation_rate)
    f2 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))


def _build_fpn(features, num_channels=64, activation='swish'):
    P3_in, P4_in, P5_in, P6_in, P7_in = features

    P7_U = UpSampling2D()(P7_in)

    P6_td = Concatenate()([P6_in, P7_U])
    P6_td = Activation(activation)(P6_td)
    P6_td = _separableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P6_td)

    P6_U = UpSampling2D()(P6_td)  # 4x4 to 9x9

    P5_td = Concatenate()([P5_in, P6_U])  # 9x9
    P5_td = Activation(activation)(P5_td)
    P5_td = _separableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P5_td)

    P5_U = UpSampling2D()(P5_td)  # 9x9 to 18x18
    P4_td = Concatenate()([P4_in, P5_U])  # 18x18
    P4_td = Activation(activation)(P4_td)
    P4_td = _separableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P4_td)

    P4_U = UpSampling2D()(P4_td)  # 18x18 to 36x36
    P3_out = Concatenate()([P3_in, P4_U])
    P3_out = Activation(activation)(P3_out)
    P3_out = _separableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P3_out)
    P3_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)  # 36x36 to 18x18
    P4_out = Concatenate()([P4_in, P4_td, P3_D])
    P4_out = Activation(activation)(P4_out)
    P4_out = _separableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P4_out)

    P4_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)  # 18x18 to 9x9
    P5_out = Concatenate()([P5_in, P5_td, P4_D])
    P5_out = Activation(activation)(P5_out)
    P5_out = _separableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P5_out)

    # padding
    P5_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)  # 9x9 to 4x4

    P6_out = Concatenate()([P6_in, P6_td, P5_D])
    P6_out = Activation(activation)(P6_out)
    P6_out = _separableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P6_out)

    P6_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
    P7_out = Concatenate()([P7_in, P6_D])
    P7_out = Activation(activation)(P7_out)
    P7_out = _separableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P7_out)

    return [P3_out, P4_td, P5_td, P6_td, P7_out]



