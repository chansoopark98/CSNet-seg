# from efficientnet_v2 import *
from model.efficientnet_v2 import *
# from model.efficientnet_v2 import EfficientNetV2S
from model.resnet101 import *
from tensorflow.keras import layers
from model.fpn_model import deepLabV3Plus, SepConv_BN, DECAY, proposed, proposed_experiments

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
    # base.load_weights('./checkpoints/efficientnetv2-s-imagenet.h5', by_name=True)")
    # base = EfficientNetV2M(input_shape=input_shape, pretrained="imagenet")

    base.summary()

    c5 = base.get_layer('add_34').output  # 16x32 256 or get_layer('post_swish') => 확장된 채널 1280
    # c5 = base.get_layer('post_swish').output  # 32x64 256 or get_layer('post_swish') => 확장된 채널 1280
    # c4 = base.get_layer('add_20').output  # 32x64 64
    # c3 = base.get_layer('add_7').output  # 64x128 48
    # c2 = base.get_layer('add_6').output  # 128x256 48
    c2 = base.get_layer('add_4').output  # 128x256 48

    """
    for EfficientNetV2M
    32x64 = 'add_50'
    64x128 = 'add_10'
    128x256 = 'add_6'
    
    'add_10'
    """

    """
    for EfficientNetV2S
    32x64 = 'add_34'
    64x128 = 'add_7'
    128x256 = 'add_4'
    """
    features = [c2, c5]

    model_input = base.input
    # model_output, aspp_aux = deepLabV3Plus(features=features, fpn_times=2, activation='swish', mode='deeplabv3+')
    model_output, aspp_aux, skip_aux = proposed(features=features, fpn_times=2, activation='relu', mode='deeplabv3+')
    # decoder_output, edge_output = proposed_experiments(features=features, activation='relu')
    """
    model_output: 128x256
    aspp_aux: 64x128
    dec_aux: 128x256"""

    total_cls = classifier(model_output, num_classes=classes, upper=4, name='output')
    # edge_cls = edge_classifier(edge_output, upper=2, name='edge')
    # body_cls = classifier(body_output, num_classes=classes, upper=4, name='body')
    aspp_aux_output = classifier(aspp_aux, num_classes=classes, upper=4, name='aspp')
    skip_aux_output = classifier(skip_aux, num_classes=classes, upper=4, name='skip')

    """
    Best Method
    Backbone : EfficientNetV2S 
        strides : [1, 2, 2, 2, 1, 1]
    
    Decoder : DeepLabV3+
    Aux : 
        EfficientV2S backbone get_layer('add_7') 64x128 (8배 업스케일링) loss factor = 0.2
        DeepLabV3+ ASPP output 4배 업스케일링 직후 loss factor = 0.5
    Learning rate 0.001
    Weight Decay = l2 (0.0001)/2
    Optimizer : Adam
    Epochs: 120
    """

    model_output = [total_cls, aspp_aux_output, skip_aux_output]

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

def classifier(x, num_classes=19, upper=4, name=None):
    x = layers.Conv2D(num_classes, 1, strides=1,
                      kernel_regularizer=DECAY,
                      kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = layers.UpSampling2D(size=(upper, upper), interpolation='bilinear', name=name)(x)
    return x

def edge_classifier(x, upper=2, name=None):
    x = layers.UpSampling2D(size=(upper, upper), interpolation='bilinear', name=name)(x)
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