# from efficientnet_v2 import *
from model.efficientnet_v2 import *
from model.ResNest import resnest
# from model.efficientnet_v2 import EfficientNetV2S
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Activation, BatchNormalization, LeakyReLU,
    GlobalAveragePooling2D, Conv2D, Dropout, Concatenate,  Add,
    DepthwiseConv2D, Reshape, ZeroPadding2D, Subtract, Flatten)
import tensorflow as tf


"""Global hyper-parameters setup"""
MOMENTUM = 0.9
EPSILON = 1e-3
# DECAY = tf.keras.regularizers.L2(l2=0.0001/2)
DECAY = None
# BN = tf.keras.layers.experimental.SyncBatchNormalization
BN = BatchNormalization
CONV_KERNEL_INITIALIZER = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_out", distribution="truncated_normal")
# CONV_KERNEL_INITIALIZER = 'he_normal'
atrous_rates= (6, 12, 18)

def deepLabV3Plus(features, activation='relu'):
    skip1, x = features # c1 48 / c2 64

    # Image Feature branch
    shape_before = tf.shape(x)
    b4 = GlobalAveragePooling2D()(x)
    b4_shape = tf.keras.backend.int_shape(b4)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Reshape((1, 1, b4_shape[1]))(b4)
    b4 = Conv2D(256, (1, 1), padding='same',
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                use_bias=False, name='image_pooling')(b4)
    b4 = BN(name='image_pooling_BN', epsilon=EPSILON, momentum=MOMENTUM)(b4)
    b4 = Activation(activation)(b4)
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    b4 = tf.keras.layers.experimental.preprocessing.Resizing(
            *size_before[1:3], interpolation="bilinear"
        )(b4)

    # b4 = UpSampling2D(size=(32, 64), interpolation="bilinear")(b4)
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same',
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                use_bias=False, name='aspp0')(x)
    # b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = BN(name='aspp0_BN', epsilon=EPSILON, momentum=MOMENTUM)(b0)
    b0 = Activation(activation, name='aspp0_activation')(b0)

    b1 = conv3x3(x, 256, 'aspp1',
                    rate=atrous_rates[0], depth_activation=True, epsilon=EPSILON, momentum=MOMENTUM, activation=activation, mode='sep')
    # rate = 12 (24)
    b2 = conv3x3(x, 256, 'aspp2',
                    rate=atrous_rates[1], depth_activation=True, epsilon=EPSILON, momentum=MOMENTUM, activation=activation, mode='sep')
    # rate = 18 (36)
    b3 = conv3x3(x, 256, 'aspp3',
                    rate=atrous_rates[2], depth_activation=True, epsilon=EPSILON, momentum=MOMENTUM, activation=activation, mode='sep')

    # concatenate ASPP branches & project
    x = Concatenate()([b4, b0, b1, b2, b3])

    x = Conv2D(256, (1, 1), padding='same',
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               use_bias=False, name='concat_projection')(x)
    # x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = BN(name='concat_projection_BN', epsilon=EPSILON, momentum=MOMENTUM)(x)
    x = Activation(activation)(x)

    x = Dropout(0.5)(x)

    skip_size = tf.keras.backend.int_shape(skip1)
    x = tf.keras.layers.experimental.preprocessing.Resizing(
        *skip_size[1:3], interpolation="bilinear"
    )(x)

    # x = UpSampling2D((4,4), interpolation='bilinear')(x)

    dec_skip1 = Conv2D(48, (1, 1), padding='same',
                       kernel_initializer=CONV_KERNEL_INITIALIZER,
                       use_bias=False, name='feature_projection0')(skip1)
    # dec_skip1 = BatchNormalization(
    #     name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = BN(
        name='feature_projection0_BN', epsilon=EPSILON, momentum=MOMENTUM)(dec_skip1)
    dec_skip1 = Activation(activation)(dec_skip1)
    x = Concatenate()([x, dec_skip1])
    x = conv3x3(x, 256, 'decoder_conv0',
                   depth_activation=True, epsilon=EPSILON, momentum=MOMENTUM, activation=activation, mode='sep')
    x = conv3x3(x, 256, 'decoder_conv1',
                   depth_activation=True, epsilon=EPSILON, momentum=MOMENTUM, activation=activation, mode='sep')

    return x

def proposed_experiments(features, activation='swish'):
    """
    1030 proposed
    miou 80%
    """
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
    b4 = BN(name='image_pooling_BN', epsilon=EPSILON, momentum=MOMENTUM)(b4)
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
    b0 = BN(name='aspp0_BN', epsilon=EPSILON, momentum=MOMENTUM)(b0)
    b0 = Activation(activation, name='aspp0_activation')(b0)

    b1 = conv3x3(x, 256, 'aspp1',
                    rate=atrous_rates[0], epsilon=EPSILON, momentum=MOMENTUM, activation=activation)
    # rate = 12 (24)
    b2 = conv3x3(x, 256, 'aspp2',
                    rate=atrous_rates[1], epsilon=EPSILON, momentum=MOMENTUM, activation=activation)
    # rate = 18 (36)
    b3 = conv3x3(x, 256, 'aspp3',
                    rate=atrous_rates[2], epsilon=EPSILON, momentum=MOMENTUM, activation=activation)
    # concatenate ASPP branches & project
    x = Concatenate()([b4, b0, b1, b2, b3])

    x = Conv2D(256, (1, 1), padding='same',
               kernel_regularizer=DECAY,
               use_bias=False, name='concat_projection')(x)
    # x = BatchNormalization(name='concat_projection_BN', epsilon=EPSILON)(x)
    x = BN(name='concat_projection_BN', epsilon=EPSILON, momentum=MOMENTUM)(x)
    x = Activation(activation)(x)

    x = Dropout(0.5)(x)

    # x to 128x256 size
    skip_size = tf.keras.backend.int_shape(skip1)
    x = tf.keras.layers.experimental.preprocessing.Resizing(
        *skip_size[1:3], interpolation="bilinear"
    )(x)

    dec_skip = Conv2D(48, (1, 1), padding='same',
                      kernel_regularizer=DECAY,
                      use_bias=False)(skip1)
    dec_skip = BN(epsilon=EPSILON, momentum=MOMENTUM)(dec_skip)
    dec_skip = Activation(activation)(dec_skip)

    edge = edge_creater(skip=dec_skip, aspp_feature=x, activation=activation)

    body = x

    x = Add()([x, edge])
    x = conv3x3(x, 256, 'edge_conv1', epsilon=EPSILON, activation=activation)

    aspp_aux = x

    x = Concatenate()([x, dec_skip])
    x = conv3x3(x, 256, 'edge_conv2', epsilon=EPSILON, activation=activation)
    x = conv3x3(x, 256, 'edge_conv3', epsilon=EPSILON, activation=activation)

    return x, edge, body, aspp_aux



def csnet_seg_model(backbone='efficientV2-s', input_shape=(512, 1024, 3), classes=19, OS=16):
    base = EfficientNetV2S(input_shape=input_shape, pretrained="imagenet")
    # base.load_weights('./checkpoints/efficientnetv2-s-imagenet.h5', by_name=True)")
    # base = EfficientNetV2M(input_shape=input_shape, pretrained="imagenet")

    base.summary()

    c5 = base.get_layer('add_34').output  # 16x32 256 or get_layer('post_swish') => 확장된 채널 1280
    # c5 = base.get_layer('post_swish').output  # 32x64 256 or get_layer('post_swish') => 확장된 채널 1280
    # c4 = base.get_layer('add_20').output  # 32x64 64
    c3 = base.get_layer('add_7').output  # 64x128 48
    # c2 = base.get_layer('add_6').output  # 128x256 48
    c2 = base.get_layer('add_4').output  # 128x256 48
    """
    for EfficientNetV2S (input resolution: 512x1024)
    32x64 = 'add_34'
    64x128 = 'add_7'
    128x256 = 'add_4'
    """
    features = [c2, c5]

    model_input = base.input
    # model_output, aspp_aux = deepLabV3Plus(features=features, fpn_times=2, activation='swish', mode='deeplabv3+')
    decoder_output, edge, body, aux = proposed_experiments(features=features, activation='swish')

    total_cls = classifier(decoder_output, num_classes=classes, upper=4, name='output')
    edge_cls = edge_classifier(edge, upper=4, name='edge')
    body_cls = classifier(body, num_classes=classes, upper=4, name='body')
    aux_cls = classifier(aux, num_classes=classes, upper=8, name='aux')


    model_output = [total_cls, edge_cls, body_cls, aux_cls]

    return model_input, model_output


def build_generator(input_shape=(512, 1024, 3), classes=19):
    # base = resnest.resnest50(input_shape=input_shape, include_top=False, input_tensor=None)
    #
    # model_input = base.input
    # c2 = base.get_layer('stage1_block3_shorcut_act').output  # 1/4 @ 256 64x64
    # x = base.get_layer('stage4_block3_shorcut_act').output  # 1/32 @2048 8x8
    # features = [c2, x]

    base = EfficientNetV2S(input_shape=input_shape, pretrained="imagenet")

    model_input = base.input
    c5 = base.get_layer('add_34').output  # 16x32 256 or get_layer('post_swish') => 확장된 채널 1280
    # c5 = base.get_layer('post_swish').output  # 32x64 256 or get_layer('post_swish') => 확장된 채널 1280
    # c4 = base.get_layer('add_20').output  # 32x64 64
    # c3 = base.get_layer('add_7').output  # 64x128 48
    # c2 = base.get_layer('add_6').output  # 128x256 48
    c2 = base.get_layer('add_4').output  # 128x256 48
    """
    for EfficientNetV2S (input resolution: 512x1024)
    32x64 = 'add_34'
    64x128 = 'add_7'
    128x256 = 'add_4'
    """
    features = [c2, c5]


    model_output = deepLabV3Plus(features=features, activation='swish')
    # decoder_output, _, _, _ = proposed_experiments(features=features, activation='swish')

    model_output = classifier(model_output, num_classes=classes, upper=4, name='output')

    return model_input, model_output

def build_discriminator(image_size=(512, 1024, 3), name='discriminator'):
    inputs = Input(shape=image_size)

    conv1 = create_conv(32, (3, 3), inputs, 'conv1', activation='leakyrelu', dropout=0., bn=False, stride=2) # 256x256
    conv2 = create_conv(64, (3, 3), conv1, 'conv2', activation='leakyrelu', dropout=.4, bn=True, stride=2) # 128x128
    conv3 = create_conv(128, (3, 3), conv2, 'conv3', activation='leakyrelu', dropout=.4, bn=True, stride=2) # 64x64
    conv4 = create_conv(256, (3, 3), conv3, 'conv4', activation='leakyrelu', dropout=.4,  bn=True, stride=2) # 32x32
    conv5 = create_conv(512, (3, 3), conv4, 'conv5', activation='leakyrelu', dropout=.4,  bn=True, stride=2) # 16x16

    flat = Flatten()(conv5)
    dense6 = Dense(1, activation='sigmoid')(flat)

    return inputs, dense6

def create_conv(filters, kernel_size, inputs, name=None, bn=True, bn_momentum=0.8,
                dropout=0., padding='same', activation='relu', stride=1):
    if bn == True:
        bias = False
    else:
        bias = True

    conv = Conv2D(filters, kernel_size, padding=padding, strides=stride, use_bias=bias,
                  kernel_initializer='he_normal', name=name)(inputs)

    if bn:
        conv = BatchNormalization(momentum=bn_momentum)(conv)

    if activation == 'leakyrelu':
        conv = LeakyReLU(alpha=0.2)(conv)
    else:
        conv = Activation(activation)(conv)

    if dropout != 0:
        conv = Dropout(dropout)(conv)

    return conv

def classifier(x, num_classes=19, upper=4, name=None):
    x = layers.Conv2D(num_classes, 1, strides=1,
                      kernel_regularizer=DECAY,
                      kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = layers.UpSampling2D(size=(upper, upper), interpolation='bilinear', name=name)(x)
    return x

def edge_classifier(x, upper=4, name=None):
    x = layers.Conv2D(1, 1, strides=1,
                      kernel_regularizer=DECAY,
                      kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = Activation('sigmoid')(x)
    x = layers.UpSampling2D(size=(upper, upper), interpolation='bilinear', name=name)(x)
    return x

def conv3x3(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3,
            momentum=0.99, activation='swish', mode='sep'):
    # convolution module function (standard convolution and depthwise-seperable convolution)
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
        x = BN(name=prefix + '_depthwise_BN', epsilon=epsilon, momentum=momentum)(x)
        if depth_activation:
            x = Activation(activation)(x)
        x = Conv2D(filters, (1, 1), padding='same',
                   kernel_regularizer=DECAY,
                   use_bias=False, name=prefix + '_pointwise')(x)
        # x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
        x = BN(name=prefix + '_pointwise_BN', epsilon=epsilon, momentum=momentum)(x)
        if depth_activation:
            x = Activation(activation)(x)

    else:
        x = Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), strides=(stride, stride),
                   padding='same', kernel_initializer=CONV_KERNEL_INITIALIZER,
               use_bias=False, dilation_rate=(rate, rate), name=prefix + '_stdConv')(x)

        # x = BatchNormalization(name=prefix + '_stdConv_BN', epsilon=epsilon)(x)
        x = BN(name=prefix + '_stdConv_BN', epsilon=epsilon, momentum=momentum)(x)
        x = Activation(activation)(x)

    return x


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3, momentum=0.99):
    activation = 'swish'
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
    x = BN(name=prefix + '_depthwise_BN', epsilon=epsilon, momentum=momentum)(x)
    if depth_activation:
        x = Activation(activation)(x)
    x = Conv2D(filters, (1, 1), padding='same',
               kernel_regularizer=DECAY,
               use_bias=False, name=prefix + '_pointwise')(x)
    # x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    x = BN(name=prefix + '_pointwise_BN', epsilon=epsilon, momentum=momentum)(x)
    if depth_activation:
        x = Activation(activation)(x)

    return x

def edge_creater(skip, aspp_feature, activation='swish'):
    concat_feature = Concatenate()([aspp_feature, skip])

    concat_feature = Conv2D(256, (1, 1), padding='same',
               kernel_regularizer=DECAY,
               use_bias=False)(concat_feature)
    concat_feature = BN(epsilon=EPSILON)(concat_feature)
    concat_feature = Activation(activation)(concat_feature)

    concat_feature = conv3x3(concat_feature, 256, prefix='aspp_feature_conv1_for_edge', stride=1, kernel_size=3, rate=1,
                             epsilon=EPSILON, activation=activation)
    concat_feature = conv3x3(concat_feature, 256, prefix='aspp_feature_conv2_for_edge', stride=1, kernel_size=3, rate=1,
                             epsilon=EPSILON, activation=activation)

    edge = Subtract()([aspp_feature, concat_feature])

    return edge
