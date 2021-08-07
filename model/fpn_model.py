from tensorflow.keras.layers import (
    MaxPooling2D, SeparableConv2D, UpSampling2D, Activation, Add, BatchNormalization, Conv2D, Dropout, Concatenate, multiply, add)
from functools import reduce
import tensorflow as tf

class GlobalAveragePooling2D(tf.keras.layers.GlobalAveragePooling2D):
    def __init__(self, keep_dims=False, **kwargs):
        super(GlobalAveragePooling2D, self).__init__(**kwargs)
        self.keep_dims = keep_dims

    def call(self, inputs):
        if self.keep_dims is False:
            return super(GlobalAveragePooling2D, self).call(inputs)
        else:
            return tf.keras.backend.mean(inputs, axis=[1, 2], keepdims=True)

    def compute_output_shape(self, input_shape):
        if self.keep_dims is False:
            return super(GlobalAveragePooling2D, self).compute_output_shape(input_shape)
        else:
            input_shape = tf.TensorShape(input_shape).as_list()
            return tf.TensorShape([input_shape[0], 1, 1, input_shape[3]])

    def get_config(self):
        config = super(GlobalAveragePooling2D, self).get_config()
        config['keep_dim'] = self.keep_dims
        return config


MOMENTUM = 0.99
EPSILON = 0.0001

def fpn_model(features, fpn_times=3, activation='swish'):
    c3, c4, c5 = features

    c6 = _convBlock(x=c5, num_channels=320, kernel_size=3, strides=2, name='feature_downsample_x1')
    c6 = Activation(activation)(c6)

    c7 = _convBlock(x=c6, num_channels=384, kernel_size=3, strides=2, name='feature_downsample_x1')
    c7 = Activation(activation)(c7)

    features = [c3, c4, c5, c6, c7]

    for i in range(fpn_times):
        features = _build_fpn(features, num_channels=224, activation=activation)

    x1 = features[0]
    x2 = features[1]
    x3 = features[2]
    x4 = features[3]
    x5 = features[4]

    x1 = gap_residual_block(x3, x1, activation=activation)
    x2 = gap_residual_block(x3, x2, activation=activation)
    x3 = gap_residual_block(x3, x3, activation=activation)
    x4 = gap_residual_block(x3, x4, activation=activation)
    x5 = gap_residual_block(x3, x5, activation=activation)

    x = Concatenate()([x1, x2, x3, x4, x5])
    x = _convBlock(x=x, num_channels=320, kernel_size=3, strides=1, name='refining_process')
    x = Activation(activation)(x)
    x = Dropout(rate=0.5)(x)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)

    x = Concatenate()([x, c4])

    x = _convBlock(x=x, num_channels=256, kernel_size=3, strides=1, name='refining_process')
    x = Activation(activation)(x)


    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Concatenate()([x, c3])
    x = _convBlock(x=x, num_channels=256, kernel_size=3, strides=1, name='refining_process')
    x = Activation(activation)(x)

    x = _convBlock(x=x, num_channels=256, kernel_size=3, strides=1, name='refining_process')
    x = Activation(activation)(x)
    x = Dropout(rate=0.5)(x)


    return x

def gap_residual_block(input_tensor, ref_tensor, activation='swish'):
    gap = GlobalAveragePooling2D(keep_dims=True)(ref_tensor)

    gap = Conv2D(224//8, 1,
                       activation=activation,
                       padding='same',
                       use_bias=True)(gap)

    gap = Conv2D(224, 1,
                       activation='sigmoid',
                       padding='same',
                       use_bias=True)(gap)

    x = multiply([input_tensor, gap])
    x = add([x, input_tensor])
    x = Activation(activation)(x)

    return x


def _convBlock(x, num_channels, kernel_size, strides, name, dilation_rate=1):
    x = SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                        dilation_rate=dilation_rate, use_bias=True)(x)
    # x = Conv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
    #            use_bias=True)(x)
    x = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(x)
    return x


def _separableConvBlock(num_channels, kernel_size, strides, dilation_rate=1):
    f1 = SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                         use_bias=True, dilation_rate=dilation_rate)
    # f1 = Conv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
    #                      use_bias=True, dilation_rate=dilation_rate)
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



