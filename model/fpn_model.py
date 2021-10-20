from tensorflow.keras.layers import (
    MaxPooling2D, SeparableConv2D, UpSampling2D, Activation, BatchNormalization,
    GlobalAveragePooling2D, Conv2D, Dropout, Concatenate, multiply, Add, concatenate,
    DepthwiseConv2D, Reshape, ZeroPadding2D, Dense, GlobalMaxPooling2D, Permute, Lambda, Subtract)
import tensorflow.keras.backend as K
import tensorflow as tf

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
EPSILON = 1e-5
DECAY = tf.keras.regularizers.L2(l2=0.0001/2)
# DECAY = None
BN = tf.keras.layers.experimental.SyncBatchNormalization
CONV_KERNEL_INITIALIZER = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_out", distribution="truncated_normal")
atrous_rates= (6, 12, 18)

def deepLabV3Plus(features, fpn_times=2, activation='swish', fpn_channels=64, mode='fpn'):
    skip1, x = features # c1 48 / c2 64

    # Image Feature branch
    shape_before = tf.shape(x)
    b4 = GlobalAveragePooling2D()(x)
    b4_shape = tf.keras.backend.int_shape(b4)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Reshape((1, 1, b4_shape[1]))(b4)
    b4 = Conv2D(256, (1, 1), padding='same',
                kernel_regularizer=DECAY,
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
    b0 = Conv2D(256, (1, 1), padding='same',
                kernel_regularizer=DECAY,
                use_bias=False, name='aspp0')(x)
    # b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = BN(name='aspp0_BN', epsilon=1e-5)(b0)
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
               kernel_regularizer=DECAY,
               use_bias=False, name='concat_projection')(x)
    # x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = BN(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation(activation)(x)

    x = Dropout(0.1)(x)

    skip_size = tf.keras.backend.int_shape(skip1)
    x = tf.keras.layers.experimental.preprocessing.Resizing(
        *skip_size[1:3], interpolation="bilinear"
    )(x)

    aux_temp_aspp = x

    # x = UpSampling2D((4,4), interpolation='bilinear')(x)

    dec_skip1 = Conv2D(48, (1, 1), padding='same',
                       kernel_regularizer=DECAY,
                       use_bias=False, name='feature_projection0')(skip1)
    # dec_skip1 = BatchNormalization(
    #     name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = BN(
        name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = Activation(activation)(dec_skip1)
    x = Concatenate()([x, dec_skip1])
    x = SepConv_BN(x, 256, 'decoder_conv0',
                   depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 256, 'decoder_conv1',
                   depth_activation=True, epsilon=1e-5)

    return x, aux_temp_aspp

def proposed(features, fpn_times=2, activation='swish', fpn_channels=64, mode='fpn'):
    skip1, x = features # c1 48 / c2 64

    # Image Feature branch
    shape_before = tf.shape(x)
    b4 = GlobalAveragePooling2D()(x)
    b4_shape = tf.keras.backend.int_shape(b4)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Reshape((1, 1, b4_shape[1]))(b4)
    b4 = Conv2D(256, (1, 1), padding='same',
                kernel_regularizer=DECAY,
                use_bias=False, name='image_pooling')(b4)
    # b4 = BatchNormalization(name='image_pooling_BN', epsilon=EPSILON)(b4)
    b4 = BN(name='image_pooling_BN', epsilon=EPSILON)(b4)
    b4 = Activation(activation)(b4)
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    b4 = tf.keras.layers.experimental.preprocessing.Resizing(
            *size_before[1:3], interpolation="bilinear"
        )(b4)

    # b4 = UpSampling2D(size=(32, 64), interpolation="bilinear")(b4)
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same',
                kernel_regularizer=DECAY,
                use_bias=False, name='aspp0')(x)
    # b0 = BatchNormalization(name='aspp0_BN', epsilon=EPSILON)(b0)
    b0 = BN(name='aspp0_BN', epsilon=EPSILON)(b0)
    b0 = Activation(activation, name='aspp0_activation')(b0)

    b1 = conv3x3(x, 256, 'aspp1',
                    rate=atrous_rates[0], epsilon=EPSILON, activation=activation)
    # rate = 12 (24)
    b2 = conv3x3(x, 256, 'aspp2',
                    rate=atrous_rates[1], epsilon=EPSILON, activation=activation)
    # rate = 18 (36)
    b3 = conv3x3(x, 256, 'aspp3',
                    rate=atrous_rates[2], epsilon=EPSILON, activation=activation)
    # concatenate ASPP branches & project
    x = Concatenate()([b4, b0, b1, b2, b3])

    x = Conv2D(256, (1, 1), padding='same',
               kernel_regularizer=DECAY,
               use_bias=False, name='concat_projection')(x)
    # x = BatchNormalization(name='concat_projection_BN', epsilon=EPSILON)(x)
    x = BN(name='concat_projection_BN', epsilon=EPSILON)(x)
    x = Activation(activation)(x)

    x = Dropout(0.1)(x)

    # x to 128x256 size
    skip_size = tf.keras.backend.int_shape(skip1)
    x = tf.keras.layers.experimental.preprocessing.Resizing(
        *skip_size[1:3], interpolation="bilinear"
    )(x)

    aspp_aux = x

    dec_skip1 = Conv2D(48, (1, 1), padding='same',
                       kernel_regularizer=DECAY,
                       use_bias=False, name='feature_projection0')(skip1)
    # dec_skip1 = BatchNormalization(
    #     name='feature_projection0_BN', epsilon=EPSILON)(dec_skip1)
    dec_skip1 = BN(
        name='feature_projection0_BN', epsilon=EPSILON)(dec_skip1)
    dec_skip1 = Activation(activation)(dec_skip1)
    x = Concatenate()([x, dec_skip1])

    skip_aux = x

    x = conv3x3(x, 256, 'decoder_conv0', epsilon=EPSILON, activation=activation)

    x = conv3x3(x, 256, 'decoder_conv1', epsilon=EPSILON, activation=activation)

    return x, aspp_aux, skip_aux

def proposed_experiments(features, activation='swish'):
    skip1, x = features # c1 48 / c2 64

    # Image Feature branch
    shape_before = tf.shape(x)
    b4 = GlobalAveragePooling2D()(x)
    b4_shape = tf.keras.backend.int_shape(b4)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Reshape((1, 1, b4_shape[1]))(b4)
    b4 = Conv2D(256, (1, 1), padding='same',
                kernel_regularizer=DECAY,
                use_bias=False, name='image_pooling')(b4)
    # b4 = BatchNormalization(name='image_pooling_BN', epsilon=EPSILON)(b4)
    b4 = BN(name='image_pooling_BN', epsilon=EPSILON)(b4)
    b4 = Activation(activation)(b4)
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    b4 = tf.keras.layers.experimental.preprocessing.Resizing(
            *size_before[1:3], interpolation="bilinear"
        )(b4)

    # b4 = UpSampling2D(size=(32, 64), interpolation="bilinear")(b4)
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same',
                kernel_regularizer=DECAY,
                use_bias=False, name='aspp0')(x)
    # b0 = BatchNormalization(name='aspp0_BN', epsilon=EPSILON)(b0)
    b0 = BN(name='aspp0_BN', epsilon=EPSILON)(b0)
    b0 = Activation(activation, name='aspp0_activation')(b0)

    b1 = conv3x3(x, 256, 'aspp1',
                    rate=atrous_rates[0], epsilon=EPSILON, activation=activation)
    # rate = 12 (24)
    b2 = conv3x3(x, 256, 'aspp2',
                    rate=atrous_rates[1], epsilon=EPSILON, activation=activation)
    # rate = 18 (36)
    b3 = conv3x3(x, 256, 'aspp3',
                    rate=atrous_rates[2], epsilon=EPSILON, activation=activation)
    # concatenate ASPP branches & project
    x = Concatenate()([b4, b0, b1, b2, b3])

    x = Conv2D(256, (1, 1), padding='same',
               kernel_regularizer=DECAY,
               use_bias=False, name='concat_projection')(x)
    # x = BatchNormalization(name='concat_projection_BN', epsilon=EPSILON)(x)
    x = BN(name='concat_projection_BN', epsilon=EPSILON)(x)
    x = Activation(activation)(x)

    x = Dropout(0.1)(x)

    # x to 128x256 size
    skip_size = tf.keras.backend.int_shape(skip1)
    x = tf.keras.layers.experimental.preprocessing.Resizing(
        *skip_size[1:3], interpolation="bilinear"
    )(x)

    aspp_aux = x

    dec_skip1, edge = edge_creater(skip_x=skip1, epsilon=EPSILON, activation=activation)

    x = Concatenate()([x, dec_skip1])

    x = conv3x3(x, 256, 'decoder_conv0', epsilon=EPSILON, activation=activation)
    x = conv3x3(x, 256, 'decoder_conv1', epsilon=EPSILON, activation=activation)


    return x, edge, aspp_aux


def edge_creater(skip_x, epsilon=1e-3, activation='relu'):

    # point-wise conv
    edge_x = Conv2D(48, (1, 1), padding='same',
               kernel_regularizer=DECAY,
               use_bias=False)(skip_x)
    edge_x = BN(epsilon=epsilon)(edge_x)
    edge_x = Activation(activation)(edge_x)

    # depth-wise conv
    edge_x = DepthwiseConv2D((3, 3), strides=(1, 1), dilation_rate=(1, 1),
                        kernel_regularizer=DECAY,
                        padding='same', use_bias=False)(edge_x)

    edge_x = BN(epsilon=epsilon)(edge_x)

    # point-wise conv
    edge_x = Conv2D(1, (1, 1), padding='same',
               kernel_regularizer=DECAY,
               use_bias=False)(edge_x)
    edge_x = BN(epsilon=epsilon)(edge_x)
    edge_x = Activation('sigmoid')(edge_x)

    edge_map = multiply([skip_x, edge_x])

    skip_x = Concatenate()([skip_x, edge_map])

    skip_x = Conv2D(48, (1, 1), padding='same',
               kernel_regularizer=DECAY,
               use_bias=False)(skip_x)
    skip_x = BN(epsilon=epsilon)(skip_x)
    skip_x = Activation(activation)(skip_x)




    return skip_x, edge_x

def edge_guide(x, edge, epsilon=1e-3, activation='relu'):
    input_x = x

    x = multiply([x, edge])

    x = Concatenate()([x, input_x])

    x = Conv2D(256, 1, 1, padding='same', use_bias=False)(x)
    x = BN(epsilon=epsilon)(x)
    x = Activation(activation)(x)

    return x




def conv3x3(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3,  activation='swish', mode='sep'):
    if mode != 'std':
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
                            kernel_regularizer=DECAY,
                            padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
        # x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
        x = BN(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
        if depth_activation:
            x = Activation(activation)(x)
        x = Conv2D(filters, (1, 1), padding='same',
                   kernel_regularizer=DECAY,
                   use_bias=False, name=prefix + '_pointwise')(x)
        # x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
        x = BN(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
        if depth_activation:
            x = Activation(activation)(x)

    else:
        x = Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), strides=(stride, stride),
                   padding='same', kernel_regularizer=DECAY,
               use_bias=False, dilation_rate=(rate, rate), name=prefix + '_stdConv')(x)

        # x = BatchNormalization(name=prefix + '_stdConv_BN', epsilon=epsilon)(x)
        x = BN(name=prefix + '_stdConv_BN', epsilon=epsilon)(x)
        x = Activation(activation)(x)

    return x


def decoding_block(input_feature, channel_ratio=8, name=None):
    input_shape = tf.keras.backend.int_shape(input_feature)

    x = channel_attention(input_feature=input_feature, ratio=channel_ratio)
    x = spatial_attention(x)

    temp = x

    # output = Conv2D(input_shape[3], (1, 1), padding='same',
    #                        kernel_regularizer=DECAY,
    #                        use_bias=False)(x)
    # output = BatchNormalization(epsilon=EPSILON)(output)
    # output = Activation('swish')(output)
    #
    # output = Concatenate()([output, input_feature])
    #
    # output = SepConv_BN(output, input_shape[3], name,
    #                depth_activation=True, epsilon=EPSILON)

    return x, temp

def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    # channel = input_feature._keras_shape[channel_axis]

    input_shape = tf.keras.backend.int_shape(input_feature)
    channel = input_shape[3]

    shared_layer_one = Dense(channel // ratio,
                             activation='swish',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    # assert avg_pool._keras_shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    # assert avg_pool._keras_shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    # assert avg_pool._keras_shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    # assert max_pool._keras_shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    # assert max_pool._keras_shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    # assert max_pool._keras_shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature, kernel_size=7):
    cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    # assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    # assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    # assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    # assert cbam_feature._keras_shape[-1] == 1


    return multiply([input_feature, cbam_feature])


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
                        kernel_regularizer=DECAY,
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    # x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    x = BN(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation(activation)(x)
    x = Conv2D(filters, (1, 1), padding='same',
               kernel_regularizer=DECAY,
               use_bias=False, name=prefix + '_pointwise')(x)
    # x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    x = BN(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation(activation)(x)

    return x

