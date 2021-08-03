# from efficientnet_v2 import *
from model.efficientnet_v2 import *
from model.resnet101 import *
from tensorflow.keras import layers
from tensorflow.keras.layers import MaxPooling2D,SeparableConv2D, UpSampling2D
from functools import reduce


BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 0.001
activation = 'swish'
aspp_size = (32, 64)

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



def csnet_seg_model(weights='pascal_voc', input_tensor=None, input_shape=(512, 1024, 3), classes=20, OS=16):
    global aspp_size
    bn_axis = 1 if K.image_data_format() == "channels_first" else -1
    """custum resnet101"""
    # input_tensor = tf.keras.Input(shape=input_shape)
    # encoder = ResNet('ResNet101', [1, 2])
    # c2, c5 = encoder(input_tensor, ['c2', 'c5'])
    aspp_size = (input_shape[0] // 16, input_shape[1] // 16)

    # """ for resnet101 """
    # base = resnet101.ResNet101(include_top=False, input_shape=input_shape, weights='imagenet')
    # base = ResNet101(include_top=False, input_shape=input_shape, weights='imagenet')
    # base.summary()
    divide_output_stride = 4
    # # x = base.get_layer('conv4_block23_out').output
    # x = base.get_layer('conv5_block3_out').output
    # # skip1 = base.get_layer('conv2_block3_out').output
    # skip1 = base.get_layer('conv2_block3_out').output
    # # conv5_block3_out 16, 32, 2048
    # # conv3_block4_out 64, 128, 512

    """ for EfficientNetV2S """


    # efficientnetv2 small
    divide_output_stride = 4
    base = EfficientNetV2S(input_shape=input_shape, classifier_activation=None, survivals=None)
    base.load_weights('./checkpoints/efficientnetv2-s-21k-ft1k.h5', by_name=True)
    # base = EfficientNetV2M(input_shape=input_shape, classifier_activation=None, survivals=None)
    # base.load_weights('./checkpoints/efficientnetv2-m-21k-ft1k.h5', by_name=True)
    base.summary()


    # x = base.get_layer('add_34').output # 32x64
    # c5 = base.get_layer('post_swish').output # 32x64 1280
    c5 = base.get_layer('add_34').output # 32x64 256
    c4 = base.get_layer('add_7').output # 64x128 64
    c2 = base.get_layer('add_4').output # 128x256 48


    # efficientnetv2 medium
    # base = EfficientNetV2('m', input_shape=input_shape, classifier_activation=None, first_strides=1)
    # base.load_weights('checkpoints/efficientnetv2-m-21k-ft1k.h5', by_name=True)
    # x = base.get_layer('add_50').output # 32x64
    # skip1 = base.get_layer('add_9').output # 128x256



    "proposed method"
    features = [c2, c4, c5]
    features = build_fpn(features=features, num_channels=256, id=0, resize=False,bn_trainable=True)
    features = build_fpn(features=features, num_channels=256, id=1, resize=False,bn_trainable=True)
    features = build_fpn(features=features, num_channels=256, id=2, resize=False,bn_trainable=True)

    x1 = features[0]
    x2 = features[1]
    x3 = features[2]
    x4 = features[3]
    x5 = features[4]

    # x1 = SeparableConvBlock(num_channels=48, kernel_size=3, strides=2, name='x1_feature_pool')(x1)
    x1 = ConvBlock(x=x1, num_channels=48, kernel_size=3, strides=2, name='x1_feature_pool')
    x1 = Activation(activation)(x1)
    x1 = MaxPooling2D(pool_size=3, strides=2, padding='same')(x1) # 128 to 32

    # x2 = SeparableConvBlock(num_channels=64, kernel_size=3, strides=2, name='x2_feature_pool')(x2)
    x2 = ConvBlock(x=x2, num_channels=64, kernel_size=3, strides=2, name='x2_feature_pool')
    x2 = Activation(activation)(x2)

    x4 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x4)
    # x4 = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1, name='x4_feature_pool')(x4)
    x4 = ConvBlock(x=x4, num_channels=64, kernel_size=3, strides=1, name='x4_feature_pool')

    x5 = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x5)
    # x5 = SeparableConvBlock(num_channels=48, kernel_size=3, strides=1, name='x5_feature_pool')(x5)
    x5 = ConvBlock(x=x5, num_channels=48, kernel_size=3, strides=1, name='x5_feature_pool')

    x = Concatenate(out_size=aspp_size)([x1, x2, x3, x4, x5])
    # x = SeparableConvBlock(num_channels=256, kernel_size=3, strides=1, name='refining_process')(x)
    x = ConvBlock(x=x, num_channels=256, kernel_size=3, strides=1, name='refining_process')
    x = Activation(activation)(x)
    x = layers.Dropout(rate=0.5)(x)

    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    # x = SeparableConvBlock(num_channels=256, kernel_size=3, strides=1, name='up4x_sep_conv')(x)
    x = ConvBlock(x=x, num_channels=256, kernel_size=3, strides=1, name='up4x_sep_conv')
    x = Activation(activation)(x)

    x = ConvBlock(x=x, num_channels=256, kernel_size=3, strides=1, name='conv_block')
    x = Activation(activation)(x)

    x = layers.Conv2D(20, 1, strides=1)(x)
    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)


    #
    # "deeplab v3+ aspp"
    # # x = _aspp(c5, 256)
    # x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    # x = _conv_bn_relu(x, 48, 1, strides=1)
    #
    # x = Concatenate(out_size=aspp_size)([x, c2])
    # x = _conv_bn_relu(x, 256, 3, 1)
    #
    # x = _conv_bn_relu(x, 256, 3, 1)
    # x = layers.Dropout(rate=0.1)(x)
    #
    # x = layers.Conv2D(20, 1, strides=1)(x)
    # x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

    return base.input, x

def _conv_bn_relu(x, filters, kernel_size, strides=1):
    bn_axis = 1 if K.image_data_format() == "channels_first" else -1
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = layers.BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(x)
    x = layers.Activation(activation)(x)
    return x

def ConvBlock(x, num_channels, kernel_size, strides, name, dilation_rate=1):
    MOMENTUM = BATCH_NORM_DECAY
    EPSILON = BATCH_NORM_EPSILON
    x = Conv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                                use_bias=True, name=name+'/conv')(x)
    x = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=name+'/bn')(x)
    return x

def SeparableConvBlock(num_channels, kernel_size, strides, name, dilation_rate=1):
    MOMENTUM = BATCH_NORM_DECAY
    EPSILON = BATCH_NORM_EPSILON
    f1 = SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                                use_bias=True, dilation_rate=dilation_rate, name=name+'/conv')
    f2 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=name+'/bn')
    # f2 = BatchNormalization(freeze=freeze_bn, name=f'{name}/bn')
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))

def build_fpn(features, num_channels=64, id=0, resize=False, bn_trainable=True):
    MOMENTUM = BATCH_NORM_DECAY
    EPSILON = BATCH_NORM_EPSILON

    if resize:
        padding = 'valid'
    else:
        padding = 'same'

    if id == 0:
        C3, C4, C5 = features
        P3_in = C3 # 36x36
        P4_in = C4 # 18x18
        P5_in = C5 # 9x9

        P6_in = Conv2D(num_channels, kernel_size=1, padding='same', name='resample_p6/conv2d')(C5)
        P6_in = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,  trainable=bn_trainable, name='resample_p6/bn')(P6_in)

        # padding
        P6_in = MaxPooling2D(pool_size=3, strides=2, padding=padding, name='resample_p6/maxpool')(P6_in) # 4x4

        P7_in = MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p7/maxpool')(P6_in) # 2x2


        if resize:
            P7_U = tf.image.resize(P7_in, (P6_in.shape[1:3]))
        else:
            P7_U = UpSampling2D()(P7_in) # 2x2 to 4x4

        P6_td = Add(name='fpn_cells/cell_/fnode0/add')([P6_in, P7_U])
        P6_td = Activation(lambda x: tf.nn.swish(x))(P6_td)
        P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name='fpn_cells/cell_/fnode0/op_after_combine5')(P6_td)
        P5_in_1 = Conv2D(num_channels, kernel_size=1, padding='same',
                                name='fpn_cells/cell_/fnode1/resample_0_2_6/conv2d')(P5_in)
        P5_in_1 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, trainable=bn_trainable,
                                            name='fpn_cells/cell_/fnode1/resample_0_2_6/bn')(P5_in_1)

        if resize:
            P6_U = tf.image.resize(P6_td, (P5_in_1.shape[1:3]))
        else:
            P6_U = UpSampling2D()(P6_td)

        P5_td = Add(name='fpn_cells/cell_/fnode1/add')([P5_in_1, P6_U]) # 9x9
        P5_td = Activation(lambda x: tf.nn.swish(x))(P5_td)
        P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name='fpn_cells/cell_/fnode1/op_after_combine6')(P5_td)
        P4_in_1 = Conv2D(num_channels, kernel_size=1, padding='same',
                                name='fpn_cells/cell_/fnode2/resample_0_1_7/conv2d')(P4_in) # 18x18
        P4_in_1 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, trainable=bn_trainable,
                                            name='fpn_cells/cell_/fnode2/resample_0_1_7/bn')(P4_in_1)

        P5_U = UpSampling2D()(P5_td)
        P4_td = Add(name='fpn_cells/cell_/fnode2/add')([P4_in_1, P5_U]) # 18x18
        P4_td = Activation(lambda x: tf.nn.swish(x))(P4_td)
        P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name='fpn_cells/cell_/fnode2/op_after_combine7')(P4_td)
        P3_in = Conv2D(num_channels, kernel_size=1, padding='same',
                              name='fpn_cells/cell_/fnode3/resample_0_0_8/conv2d')(P3_in) # 36x36
        P3_in = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, trainable=bn_trainable,
                                          name=f'fpn_cells/cell_/fnode3/resample_0_0_8/bn')(P3_in)

        P4_U = UpSampling2D()(P4_td) # 18x18 to 36x36
        P3_out = Add(name='fpn_cells/cell_/fnode3/add')([P3_in, P4_U])
        P3_out = Activation(lambda x: tf.nn.swish(x))(P3_out)
        P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name='fpn_cells/cell_/fnode3/op_after_combine8')(P3_out)
        P4_in_2 = Conv2D(num_channels, kernel_size=1, padding='same',
                                name='fpn_cells/cell_/fnode4/resample_0_1_9/conv2d')(P4_in)
        P4_in_2 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, trainable=bn_trainable,
                                            name='fpn_cells/cell_/fnode4/resample_0_1_9/bn')(P4_in_2)

        P3_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
        P4_out = Add(name='fpn_cells/cell_/fnode4/add')([P4_in_2, P4_td, P3_D])
        P4_out = Activation(lambda x: tf.nn.swish(x))(P4_out)
        P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name='fpn_cells/cell_/fnode4/op_after_combine9')(P4_out)

        P5_in_2 = Conv2D(num_channels, kernel_size=1, padding='same',
                                name='fpn_cells/cell_/fnode5/resample_0_2_10/conv2d')(P5_in)
        P5_in_2 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, trainable=bn_trainable,
                                            name='fpn_cells/cell_/fnode5/resample_0_2_10/bn')(P5_in_2)

        P4_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
        P5_out = Add(name='fpn_cells/cell_/fnode5/add')([P5_in_2, P5_td, P4_D]) # 9x9
        P5_out = Activation(lambda x: tf.nn.swish(x))(P5_out)
        P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name='fpn_cells/cell_/fnode5/op_after_combine10')(P5_out)

        # padding
        P5_D = MaxPooling2D(pool_size=3, strides=2, padding=padding)(P5_out) # 9x9 to 4x4

        P6_out = Add(name='fpn_cells/cell_/fnode6/add')([P6_in, P6_td, P5_D])
        P6_out = Activation(lambda x: tf.nn.swish(x))(P6_out)
        P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name='fpn_cells/cell_/fnode6/op_after_combine11')(P6_out)

        P6_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
        P7_out = Add(name='fpn_cells/cell_/fnode7/add')([P7_in, P6_D])
        P7_out = Activation(lambda x: tf.nn.swish(x))(P7_out)
        P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name='fpn_cells/cell_/fnode7/op_after_combine12')(P7_out)


        return [P3_out, P4_td, P5_td, P6_td, P7_out]

    else:

        P3_in, P4_in, P5_in, P6_in, P7_in = features

        if resize:
            P7_U = tf.image.resize(P7_in, (P6_in.shape[1:3]))
        else:
            P7_U = UpSampling2D()(P7_in) # 2x2 to 4x4

        P6_td = Add(name=f'fpn_cells/cell_{id}/fnode0/add')([P6_in, P7_U])
        P6_td = Activation(lambda x: tf.nn.swish(x))(P6_td)
        P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)

        if resize:
            P6_U = tf.image.resize(P6_td, (P5_in.shape[1:3]))
        else:
            P6_U = UpSampling2D()(P6_td) # 4x4 to 9x9

        P5_td = Add(name=f'fpn_cells/cell_{id}/fnode1/add')([P5_in, P6_U]) # 9x9
        P5_td = Activation(lambda x: tf.nn.swish(x))(P5_td)
        P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)
        P5_U = UpSampling2D()(P5_td) # 9x9 to 18x18
        P4_td = Add(name=f'fpn_cells/cell_{id}/fnode2/add')([P4_in, P5_U]) # 18x18
        P4_td = Activation(lambda x: tf.nn.swish(x))(P4_td)
        P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')(P4_td)
        P4_U = UpSampling2D()(P4_td) # 18x18 to 36x36
        P3_out = Add(name=f'fpn_cells/cell_{id}/fnode3/add')([P3_in, P4_U])
        P3_out = Activation(lambda x: tf.nn.swish(x))(P3_out)
        P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')(P3_out)
        P3_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out) # 36x36 to 18x18
        P4_out = Add(name=f'fpn_cells/cell_{id}/fnode4/add')([P4_in, P4_td, P3_D])
        P4_out = Activation(lambda x: tf.nn.swish(x))(P4_out)
        P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')(P4_out)

        P4_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out) # 18x18 to 9x9
        P5_out = Add(name=f'fpn_cells/cell_{id}/fnode5/add')([P5_in, P5_td, P4_D])
        P5_out = Activation(lambda x: tf.nn.swish(x))(P5_out)
        P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')(P5_out)

        # padding
        P5_D = MaxPooling2D(pool_size=3, strides=2, padding=padding)(P5_out)  # 9x9 to 4x4

        P6_out = Add(name=f'fpn_cells/cell_{id}/fnode6/add')([P6_in, P6_td, P5_D])
        P6_out = Activation(lambda x: tf.nn.swish(x))(P6_out)
        P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')(P6_out)

        P6_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
        P7_out = Add(name=f'fpn_cells/cell_{id}/fnode7/add')([P7_in, P6_D])
        P7_out = Activation(lambda x: tf.nn.swish(x))(P7_out)
        P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')(P7_out)

        return [P3_out, P4_td, P5_td, P6_td, P7_out]

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