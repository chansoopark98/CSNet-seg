import tensorflow.keras.regularizers
from tensorflow.keras.layers import (
    MaxPooling2D, SeparableConv2D, UpSampling2D, Activation, BatchNormalization,
    GlobalAveragePooling2D, Conv2D, Dropout, Concatenate, multiply, Add, concatenate,
    DepthwiseConv2D, Reshape, ZeroPadding2D)
from functools import reduce
import tensorflow as tf
#
# class GlobalAveragePooling2D(tf.keras.layers.GlobalAveragePooling2D):
#     def __init__(self, keep_dims=False, **kwargs):
#         super(GlobalAveragePooling2D, self).__init__(**kwargs)
#         self.keep_dims = keep_dims
#
#     def call(self, inputs):
#         if self.keep_dims is False:
#             return super(GlobalAveragePooling2D, self).call(inputs)
#         else:
#             return tf.keras.backend.mean(inputs, axis=[1, 2], keepdims=True)
#
#     def compute_output_shape(self, input_shape):
#         if self.keep_dims is False:
#             return super(GlobalAveragePooling2D, self).compute_output_shape(input_shape)
#         else:
#             input_shape = tf.TensorShape(input_shape).as_list()
#             return tf.TensorShape([input_shape[0], 1, 1, input_shape[3]])
#
#     def get_config(self):
#         config = super(GlobalAveragePooling2D, self).get_config()
#         config['keep_dim'] = self.keep_dims
#         return config


MOMENTUM = 0.99
EPSILON = 1e-3
# DECAY = tf.keras.regularizers.L2(l2=0.00001//2)
DECAY = None
CONV_KERNEL_INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
atrous_rates= (6, 12, 18)

def fpn_model(features, fpn_times=2, activation='swish', fpn_channels=256):
    skip1, x = features # c1 48 / c2 64

    # Image Feature branch
    shape_before = tf.shape(x)
    b4 = GlobalAveragePooling2D()(x)
    b4_shape = tf.keras.backend.int_shape(b4)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Reshape((1, 1, b4_shape[1]))(b4)
    b4 = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation(activation)(b4)
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    b4 = tf.keras.layers.experimental.preprocessing.Resizing(
            *size_before[1:3], interpolation="bilinear"
        )(b4)

    # b4 = UpSampling2D(size=(32, 64), interpolation="bilinear")(b4)
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation(activation, name='aspp0_activation')(b0)

    b1 = SepConv_BN(x, 256, 'aspp1',
                    rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    # rate = 12 (24)
    b2 = SepConv_BN(x, 256, 'aspp2',
                    rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    # rate = 18 (36)
    b3 = SepConv_BN(x, 256, 'aspp3',
                    rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

    # concatenate ASPP branches & project
    x = Concatenate()([b4, b0, b1, b2, b3])

    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation(activation)(x)
    x = Dropout(0.1)(x)

    skip_size = tf.keras.backend.int_shape(skip1)
    x = tf.keras.layers.experimental.preprocessing.Resizing(
        *skip_size[1:3], interpolation="bilinear"
    )(x)

    # x = UpSampling2D((4,4), interpolation='bilinear')(x)

    dec_skip1 = Conv2D(48, (1, 1), padding='same',
                       use_bias=False, name='feature_projection0')(skip1)
    dec_skip1 = BatchNormalization(
        name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = Activation(activation)(dec_skip1)
    x = Concatenate()([x, dec_skip1])
    x = SepConv_BN(x, 256, 'decoder_conv0',
                   depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 256, 'decoder_conv1',
                   depth_activation=True, epsilon=1e-5)


    return x


def gap_residual_block(input_tensor, ref_tensor, activation='swish', fpn_channels=224, squeeze_ratio=8):
    gap = GlobalAveragePooling2D(keep_dims=True)(ref_tensor)
    # gap = tf.reduce_mean(ref_tensor, [1, 2], keepdims=True) # test for


    gap = Conv2D(fpn_channels//squeeze_ratio, 1,
                       activation=activation,
                       padding='same',
                       use_bias=True)(gap)

    gap = Conv2D(fpn_channels, 1,
                       activation='sigmoid',
                       padding='same',
                       use_bias=True)(gap)

    x = multiply([input_tensor, gap])

    x = concatenate([x, input_tensor])

    x = SeparableConv2D(fpn_channels, 3, 1, padding='same', kernel_initializer=CONV_KERNEL_INITIALIZER,
               use_bias=False, kernel_regularizer=DECAY)(x)
    x = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(x)
    x = Activation(activation)(x)

    return x


def UpseperableConv(input_tensor, filters, kernel_size, strides, dilation_rate, activation):
    x = UpSampling2D(interpolation='bilinear')(input_tensor)
    x = SeparableConv2D(filters, kernel_size, strides, padding='same', use_bias=False, dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    return x

def seperableConv(input_tensor, filters, kernel_size, strides, dilation_rate, activation):
    x = SeparableConv2D(filters, kernel_size, strides, padding='same', use_bias=False,
                        dilation_rate=dilation_rate)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    return x



def _convBlock(input_tensor, output_filter, expand_ratio, kernel_size, strides, use_ca=True, dilation_rate=1, activation='swish'):
    # x = SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same', kernel_regularizer=DECAY,
    #                     kernel_initializer=CONV_KERNEL_INITIALIZER, use_bias=True, dilation_rate=dilation_rate)(x)
    # x = Conv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
    #            use_bias=True)(x)
    # x = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(x)

    input_channels = input_tensor.shape[3]


    filters = input_channels * expand_ratio

    # 채널 확장
    x = Conv2D(filters, 1,
               padding='same',
               use_bias=False,
               kernel_initializer=CONV_KERNEL_INITIALIZER)(input_tensor)

    x = BatchNormalization(axis=3)(x)
    x = Activation(activation)(x)

    # Depth-wise Convolution 페이즈
    x = DepthwiseConv2D(kernel_size,
                        strides=strides,
                        padding='same',
                        use_bias='False',
                        depthwise_initializer=CONV_KERNEL_INITIALIZER,
                        dilation_rate=dilation_rate)(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation(activation)(x)

    if use_ca:
        num_reduced_filters = max(1, int(input_channels * 0.25))
        ca_tensor = GlobalAveragePooling2D()(x)

        target_shape = (1, 1, filters) if tf.keras.backend.image_data_format() == 'channels_last' else (filters, 1, 1)
        ca_tensor = Reshape(target_shape)(ca_tensor)
        ca_tensor = Conv2D(num_reduced_filters, 1,
                           activation=activation,
                           padding='same',
                           use_bias=True,
                           kernel_initializer=CONV_KERNEL_INITIALIZER,
                           )(ca_tensor)
        ca_tensor = Conv2D(filters, 1,
                           activation='sigmoid',
                           padding='same',
                           use_bias=True,
                           kernel_initializer=CONV_KERNEL_INITIALIZER,
                           )(ca_tensor)
        x = multiply([x, ca_tensor])



    # 채널 확장
    x = Conv2D(output_filter, 1,
               padding='same',
               use_bias=False,
               kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = BatchNormalization(axis=3)(x)

    # 입력 특징 잔차
    res_x = Conv2D(output_filter, 1,
               padding='same',
               use_bias=False,
               kernel_initializer=CONV_KERNEL_INITIALIZER)(input_tensor)
    res_x = BatchNormalization(axis=3)(res_x)

    x = Add()([x, res_x])
    x = Activation(activation)(x)

    return x

def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    activation = 'swish'
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation(activation)(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation(activation)(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation(activation)(x)

    return x

def _separableConvBlock(num_channels, kernel_size, strides, dilation_rate=1):
    f1 = SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same', kernel_regularizer=DECAY,
                         kernel_initializer=CONV_KERNEL_INITIALIZER,
                         use_bias=False, dilation_rate=dilation_rate)
    # f1 = Conv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
    #                      use_bias=True, dilation_rate=dilation_rate)
    f2 = BatchNormalization()
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))


def _build_fpn(features, num_channels=64, id=0, activation='swish'):
    padding = 'same'

    if id == 0:
        C4, C5= features
        # P3_in = C3
        P4_in = C4
        P5_in = C5
        # P6_in = C6

        P6_in = Conv2D(num_channels, kernel_size=1, padding='same', name='resample_p6/conv2d')(C5)
        P6_in = BatchNormalization(name='resample_p6/bn')(P6_in)

        # padding
        P6_in = MaxPooling2D(pool_size=3, strides=2, padding=padding, name='resample_p6/maxpool')(P6_in)

        P6_td = Activation(activation)(P6_in)
        P6_td = _separableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P6_td)

        P5_in_1 = Conv2D(num_channels, kernel_size=1, padding='same',
                                name='fpn_cells/cell_/fnode1/resample_0_2_6/conv2d')(P5_in)
        P5_in_1 = BatchNormalization(
                                            name='fpn_cells/cell_/fnode1/resample_0_2_6/bn')(P5_in_1)

        P6_U = UpSampling2D()(P6_td)

        P5_td = Add(name='fpn_cells/cell_/fnode1/add')([P5_in_1, P6_U]) # 9x9
        P5_td = Activation(activation)(P5_td)
        P5_td = _separableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P5_td)
        P4_in_1 = Conv2D(num_channels, kernel_size=1, padding='same',
                                name='fpn_cells/cell_/fnode2/resample_0_1_7/conv2d')(P4_in) # 18x18
        P4_in_1 = BatchNormalization(
                                            name='fpn_cells/cell_/fnode2/resample_0_1_7/bn')(P4_in_1)

        P5_U = UpSampling2D()(P5_td)
        P4_td = Add(name='fpn_cells/cell_/fnode2/add')([P4_in_1, P5_U]) # 18x18
        P4_td = Activation(activation)(P4_td)
        P4_td = _separableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P4_td)

        P4_in_2 = Conv2D(num_channels, kernel_size=1, padding='same',
                                name='fpn_cells/cell_/fnode4/resample_0_1_9/conv2d')(P4_in)
        P4_in_2 = BatchNormalization(
                                            name='fpn_cells/cell_/fnode4/resample_0_1_9/bn')(P4_in_2)


        P4_out = Add(name='fpn_cells/cell_/fnode4/add')([P4_in_2, P4_td])
        P4_out = Activation(activation)(P4_out)
        P4_out = _separableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P4_out)

        P5_in_2 = Conv2D(num_channels, kernel_size=1, padding='same',
                                name='fpn_cells/cell_/fnode5/resample_0_2_10/conv2d')(P5_in)
        P5_in_2 = BatchNormalization(
                                            name='fpn_cells/cell_/fnode5/resample_0_2_10/bn')(P5_in_2)

        P4_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
        P5_out = Add(name='fpn_cells/cell_/fnode5/add')([P5_in_2, P5_td, P4_D]) # 9x9
        P5_out = Activation(activation)(P5_out)
        P5_out = _separableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P5_out)

        # padding
        P5_D = MaxPooling2D(pool_size=3, strides=2, padding=padding)(P5_out) # 9x9 to 4x4

        P6_out = Add(name='fpn_cells/cell_/fnode6/add')([P6_in, P6_td, P5_D])
        P6_out = Activation(activation)(P6_out)
        P6_out = _separableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P6_out)


        return [P4_out, P5_out, P6_out]

    else:

        P4_in, P5_in, P6_in = features

        # P7_U = UpSampling2D()(P7_in) # 2x2 to 4x4

        # P6_td = Add(name=f'fpn_cells/cell_{id}/fnode0/add')([P6_in, P7_U])
        # P6_td = Activation(activation)(P6_td)
        P6_td = _separableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P6_in)

        P6_U = UpSampling2D()(P6_td) # 4x4 to 9x9

        P5_td = Add(name=f'fpn_cells/cell_{id}/fnode1/add')([P5_in, P6_U]) # 9x9
        P5_td = Activation(activation)(P5_td)
        P5_td = _separableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P5_td)

        P5_U = UpSampling2D()(P5_td) # 9x9 to 18x18
        P4_td = Add(name=f'fpn_cells/cell_{id}/fnode2/add')([P4_in, P5_U]) # 18x18
        P4_td = Activation(activation)(P4_td)
        P4_td = _separableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P4_td)


        P4_out = Add(name=f'fpn_cells/cell_{id}/fnode4/add')([P4_in, P4_td])
        P4_out = Activation(activation)(P4_out)
        P4_out = _separableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P4_out)

        P4_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out) # 18x18 to 9x9
        P5_out = Add(name=f'fpn_cells/cell_{id}/fnode5/add')([P5_in, P5_td, P4_D])
        P5_out = Activation(activation)(P5_out)
        P5_out = _separableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P5_out)

        # padding
        P5_D = MaxPooling2D(pool_size=3, strides=2, padding=padding)(P5_out)  # 9x9 to 4x4

        P6_out = Add(name=f'fpn_cells/cell_{id}/fnode6/add')([P6_in, P6_td, P5_D])
        P6_out = Activation(activation)(P6_out)
        P6_out = _separableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P6_out)

        # P6_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
        # P7_out = Add(name=f'fpn_cells/cell_{id}/fnode7/add')([P7_in, P6_D])
        # P7_out = Activation(activation)(P7_out)
        # P7_out = _separableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P7_out)


        return [P4_out, P5_out, P6_out]

