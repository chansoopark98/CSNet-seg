# from efficientnet_v2 import *
from model.efficientnet_v2 import *
# from model.efficientnet_v2 import EfficientNetV2S
from model.resnet101 import *
from tensorflow.keras import layers
from model.fpn_model import deepLabV3Plus, SepConv_BN, DECAY, proposed

from tensorflow.keras.layers import (
    MaxPooling2D, SeparableConv2D, UpSampling2D, Activation, BatchNormalization,
    GlobalAveragePooling2D, Conv2D, Dropout, Concatenate, multiply, Add, concatenate,
    DepthwiseConv2D, Reshape, ZeroPadding2D)



CONV_KERNEL_INITIALIZER = keras.initializers.VarianceScaling(scale=1.0, mode="fan_out", distribution="truncated_normal")
BATCH_NORM_DECAY = 0.99
BATCH_NORM_EPSILON = 0.001
activation = 'swish'
aspp_size = (32, 64)

GET_EFFICIENT_NAME = {
    'B0': ['block3b_add', 'block5c_add', 'block7a_project_bn'],
    'B1': ['block3c_add', 'block5d_add', 'block7b_add'],
    'B2': ['block3c_add', 'block5d_add', 'block7b_add'],
    'B3': ['block3c_add', 'block5e_add', 'block7b_add'],
    'B4': ['block3d_add', 'block5f_add', 'block7b_add'],
    'B5': ['block3e_add', 'block5g_add', 'block7c_add'],
    'B6': ['block3f_add', 'block5h_add', 'block7c_add'],
    'B7': ['block3g_add', 'block5j_add', 'block7d_add']
}


class Concatenate(tf.keras.layers.Concatenate):
    def __init__(self, out_size=None, axis=-1, name=None):
        super(Concatenate, self).__init__(axis=axis, name=name)
        self.out_size = out_size

    def call(self, inputs):
        return tf.keras.backend.concatenate(inputs, self.axis)

    def build(self, input_shape):
        pass

    def compute_output_shape(self, input_shape):
        if self.out_size is None:
            return super(Concatenate, self).compute_output_shape(input_shape)
        else:
            if not isinstance(input_shape, list):
                raise ValueError('A `Concatenate` layer should be called '
                                 'on a list of inputs.')
            input_shapes = input_shape
            output_shape = list(input_shapes[0])
            for shape in input_shapes[1:]:
                if output_shape[self.axis] is None or shape[self.axis] is None:
                    output_shape[self.axis] = None
                    break
                output_shape[self.axis] += shape[self.axis]
            return tuple([output_shape[0]] + list(self.out_size) + [output_shape[-1]])

    def get_config(self):
        config = super(Concatenate, self).get_config()
        config['out_size'] = self.out_size
        return config



def csnet_seg_model(backbone='efficientV2-s', input_shape=(512, 1024, 3), classes=19, OS=16):
    # global aspp_size
    # bn_axis = 1 if K.image_data_format() == "channels_first" else -1
    # aspp_size = (input_shape[0] // OS, input_shape[1] // OS)

    # if backbone == 'ResNet101':
    #     input_tensor = tf.keras.Input(shape=input_shape)
    #     encoder = ResNet('ResNet101', [1, 2])
    #     c2, c5 = encoder(input_tensor, ['c2', 'c5'])
    #
    #
    # elif backbone == 'efficientV1-b0':
    #     base = efn.EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    #     base.summary()
    #     c3 = base.get_layer('block2b_add').output
    #     c4 = base.get_layer('block3b_add').output
    #     c5 = base.get_layer('block7a_project_bn').output
    #
    #     features = [c3, c4, c5]
    #
    #     model_input = base.input
    #     model_output = fpn_model(features=features, fpn_times=3, activation='swish')
    #     model_output = classifier(model_output, num_classes=classes)
    #
    # elif backbone == 'efficientV2-s':
        # base = EfficientNetV2S(input_shape=input_shape, classifier_activation=None, survivals=None)
        # base = EfficientNetV2S(pretrained="imagenet21k-ft1k", input_shape=input_shape, num_classes=0, dropout=0.2)
        # base = EfficientNetV2S(pretrained="imagenet", input_shape=input_shape, num_classes=0, dropout=1e-6)
    base = EfficientNetV2S(input_shape=input_shape, pretrained="imagenet")

    base.summary()
    base.load_weights('./checkpoints/efficientnetv2-s-imagenet.h5', by_name=True)
    c5 = base.get_layer('add_34').output  # 16x32 256 or get_layer('post_swish') => 확장된 채널 1280
    # c5 = base.get_layer('post_swish').output  # 32x64 256 or get_layer('post_swish') => 확장된 채널 1280
    # c4 = base.get_layer('add_20').output  # 32x64 64
    c3 = base.get_layer('add_7').output  # 64x128 48
    c2 = base.get_layer('add_4').output  # 128x256 48

    features = [c2, c3, c5]

    model_input = base.input
    # model_output, aspp_aux = deepLabV3Plus(features=features, fpn_times=2, activation='swish', mode='deeplabv3+')
    model_output, aspp_aux = proposed(features=features, fpn_times=2, activation='swish', mode='deeplabv3+')
    decoder_output = classifier(model_output, num_classes=classes, upper=4, name='output')
    aux_output = classifier(c3, num_classes=classes, use_aux=True, upper=8, name='aux')
    aspp_aux_output = classifier(aspp_aux, num_classes=classes, use_aux=True, upper=4, name='aspp_aux')

    model_output = [decoder_output, aux_output, aspp_aux_output]


    # elif backbone == 'efficientV2-m':
    #     base = EfficientNetV2('m', input_shape=input_shape, classifier_activation=None, first_strides=1)
    #     # base = EfficientNetV2M(input_shape=input_shape, classifier_activation=None, survivals=None)
    #     base.load_weights('checkpoints/efficientnetv2-m-21k-ft1k.h5', by_name=True)
    #     c5 = base.get_layer('add_50').output # 32x64
    #     c4 = base.get_layer('add_29').output # 128x256
    #     c3 = base.get_layer('add_10').output # 128x256
    #     # base.summary()
    #
    #     features = [c3, c4, c5]
    #     model_input = base.input
    #     model_output = fpn_model(features=features, fpn_times=3, activation='swish')
    #     model_output = classifier(model_output, num_classes=classes)


    # elif backbone =='xception':
    #
    #     base = EfficientNetV2S(input_shape=input_shape, pretrained="imagenet")
    #     img_input = base.input
    #
    #     x = base.get_layer('add_34').output  # 16x32 256 or get_layer('post_swish') => 확장된 채널 1280
    #     skip1 = base.get_layer('add_4').output  # 128x256 48
    #     #
    #     #
    #     # img_input = Input(shape=input_shape)
    #     # entry_block3_stride = 2
    #     # middle_block_rate = 1
    #     # exit_block_rates = (1, 2)
    #     atrous_rates = (6, 12, 18)
    #     #
    #     # x = Conv2D(32, (3, 3), strides=(2, 2),
    #     #            name='entry_flow_conv1_1', use_bias=False, padding='same')(img_input)
    #     # x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
    #     # x = Activation(tf.nn.relu)(x)
    #     #
    #     # x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
    #     # x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
    #     # x = Activation(tf.nn.relu)(x)
    #     #
    #     # x = _xception_block(x, [128, 128, 128], 'entry_flow_block1',
    #     #                     skip_connection_type='conv', stride=2,
    #     #                     depth_activation=False)
    #     # x, skip1 = _xception_block(x, [256, 256, 256], 'entry_flow_block2',
    #     #                            skip_connection_type='conv', stride=2,
    #     #                            depth_activation=False, return_skip=True)
    #     #
    #     # x = _xception_block(x, [728, 728, 728], 'entry_flow_block3',
    #     #                     skip_connection_type='conv', stride=entry_block3_stride,
    #     #                     depth_activation=False)
    #     # for i in range(16):
    #     #     x = _xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
    #     #                         skip_connection_type='sum', stride=1, rate=middle_block_rate,
    #     #                         depth_activation=False)
    #     #
    #     # x = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
    #     #                     skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
    #     #                     depth_activation=False)
    #     # x = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
    #     #                     skip_connection_type='none', stride=1, rate=exit_block_rates[1],
    #     #                     depth_activation=True)
    #
    #     # Image Feature branch
    #     shape_before = tf.shape(x)
    #     b4 = GlobalAveragePooling2D()(x)
    #     b4_shape = tf.keras.backend.int_shape(b4)
    #     # from (b_size, channels)->(b_size, 1, 1, channels)
    #     b4 = Reshape((1, 1, b4_shape[1]))(b4)
    #     b4 = Conv2D(256, (1, 1), padding='same',
    #                 use_bias=False, name='image_pooling')(b4)
    #     b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    #     b4 = Activation(tf.nn.relu)(b4)
    #     # upsample. have to use compat because of the option align_corners
    #     size_before = tf.keras.backend.int_shape(x)
    #     b4 = tf.keras.layers.experimental.preprocessing.Resizing(
    #         *size_before[1:3], interpolation="bilinear"
    #     )(b4)
    #     # simple 1x1
    #     b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    #     b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    #     b0 = Activation(tf.nn.relu, name='aspp0_activation')(b0)
    #
    #     b1 = SepConv_BN(x, 256, 'aspp1',
    #                     rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    #     # rate = 12 (24)
    #     b2 = SepConv_BN(x, 256, 'aspp2',
    #                     rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    #     # rate = 18 (36)
    #     b3 = SepConv_BN(x, 256, 'aspp3',
    #                     rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)
    #
    #     # concatenate ASPP branches & project
    #     x = Concatenate()([b4, b0, b1, b2, b3])
    #
    #     x = Conv2D(256, (1, 1), padding='same',
    #                use_bias=False, name='concat_projection')(x)
    #     x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    #     x = Activation(tf.nn.relu)(x)
    #     x = Dropout(0.1)(x)
    #
    #     skip_size = tf.keras.backend.int_shape(skip1)
    #     x = tf.keras.layers.experimental.preprocessing.Resizing(
    #         *skip_size[1:3], interpolation="bilinear"
    #     )(x)
    #     dec_skip1 = Conv2D(48, (1, 1), padding='same',
    #                        use_bias=False, name='feature_projection0')(skip1)
    #     dec_skip1 = BatchNormalization(
    #         name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    #     dec_skip1 = Activation(tf.nn.relu)(dec_skip1)
    #     x = Concatenate()([x, dec_skip1])
    #     x = SepConv_BN(x, 256, 'decoder_conv0',
    #                    depth_activation=True, epsilon=1e-5)
    #     x = SepConv_BN(x, 256, 'decoder_conv1',
    #                    depth_activation=True, epsilon=1e-5)
    #
    #     model_output = classifier(x, num_classes=classes)
    #     model_input = img_input
    #
    # else:
    #     raise print("Check your backbone name!")
    #
    #
    #
    #
    # #
    # # "deeplab v3+ aspp"
    # # # x = _aspp(c5, 256)
    # # x = layers.Dropout(rate=0.5)(x)
    # # x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    # # x = _conv_bn_relu(x, 48, 1, strides=1)
    # #
    # # x = Concatenate(out_size=aspp_size)([x, c2])
    # # x = _conv_bn_relu(x, 256, 3, 1)
    # # x = layers.Dropout(rate=0.5)(x)#
    # # x = _conv_bn_relu(x, 256, 3, 1)
    # # x = layers.Dropout(rate=0.1)(x)
    # #
    # # x = layers.Conv2D(20, 1, strides=1)(x)
    # # x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

    return model_input, model_output


def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    """Implements right 'same' padding for even kernel sizes
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
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
        x = Activation(tf.nn.relu)(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation(tf.nn.relu)(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation(tf.nn.relu)(x)

    return x



def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                    rate=1, depth_activation=False, return_skip=False):
    """ Basic building block of modified Xception network
        Args:
            inputs: input tensor
            depth_list: number of filters in each SepConv layer. len(depth_list) == 3
            prefix: prefix before name
            skip_connection_type: one of {'conv','sum','none'}
            stride: stride at last depthwise conv
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & pointwise convs
            return_skip: flag to return additional tensor after 2 SepConvs for decoder
            """
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                kernel_size=1,
                                stride=stride)
        shortcut = BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs

def classifier(x, num_classes=19, use_aux=False, upper=4, name=None):
    upper_factor = upper
    if use_aux:

        x = SepConv_BN(x, 256, name,
                   depth_activation=True, epsilon=1e-5)


    x = layers.Conv2D(num_classes, 1, strides=1,
                      kernel_regularizer=DECAY,
                      kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = layers.UpSampling2D(size=(upper_factor, upper_factor), interpolation='bilinear', name=name)(x)
    return x

def _conv_bn_relu(x, filters, kernel_size, strides=1):
    bn_axis = 1 if K.image_data_format() == "channels_first" else -1
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = layers.BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(x)
    x = layers.Activation(activation)(x)
    return x

def _conv_bn_relu(x, filters, kernel_size, strides=1):
    bn_axis = 1 if K.image_data_format() == "channels_first" else -1
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = layers.BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(x)
    x = layers.Activation(activation)(x)
    return x

def _aspp(x, out_filters):
    bn_axis = 1 if K.image_data_format() == "channels_first" else -1
    xs = list()
    x1 = layers.Conv2D(out_filters, 1, strides=1)(x)
    xs.append(x1)
    # aspp_size = (512//16, 1024//16)
    for i in range(3):
        xi = layers.Conv2D(out_filters, 3, strides=1, padding='same', dilation_rate=6 * (i + 1))(x)
        xs.append(xi)
    img_pool = GlobalAveragePooling2D(keep_dims=True)(x)
    img_pool = layers.Conv2D(out_filters, 1, 1, kernel_initializer='he_normal')(img_pool)
    img_pool = layers.UpSampling2D(size=aspp_size, interpolation='bilinear')(img_pool)
    xs.append(img_pool)
    x = Concatenate(out_size=aspp_size)(xs)
    x = layers.Conv2D(out_filters, 1, strides=1, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(x)

    return x