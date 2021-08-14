# from efficientnet_v2 import *
from model.efficientnet_v2 import *
from model.resnet101 import *
from tensorflow.keras import layers
from model.fpn_model import fpn_model

CONV_KERNEL_INITIALIZER = keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
BATCH_NORM_DECAY = 0.99
BATCH_NORM_EPSILON = 0.001
activation = 'swish'
aspp_size = (32, 64)




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



def csnet_seg_model(backbone='efficientV2-s', input_shape=(512, 1024, 3), classes=20, OS=16):
    # global aspp_size
    # bn_axis = 1 if K.image_data_format() == "channels_first" else -1
    # aspp_size = (input_shape[0] // OS, input_shape[1] // OS)

    if backbone == 'ResNet101':
        input_tensor = tf.keras.Input(shape=input_shape)
        encoder = ResNet('ResNet101', [1, 2])
        c2, c5 = encoder(input_tensor, ['c2', 'c5'])

    elif backbone == 'efficientV2-s':
        base = EfficientNetV2S(input_shape=input_shape, classifier_activation=None, survivals=None)
        base.load_weights('./checkpoints/efficientnetv2-s-21k-ft1k.h5', by_name=True)
        c5 = base.get_layer('add_34').output  # 32x64 256 or get_layer('post_swish') => 확장된 채널 1280
        # c5 = base.get_layer('post_swish').output  # 32x64 256 or get_layer('post_swish') => 확장된 채널 1280
        c4 = base.get_layer('add_7').output  # 64x128 64
        c3 = base.get_layer('add_4').output  # 128x256 48

        features = [c3, c4, c5]

        model_input = base.input
        model_output = fpn_model(features=features, fpn_times=3, activation='relu')
        model_output = classifier(model_output, num_classes=classes)


    elif backbone == 'efficientV2-m':
        base = EfficientNetV2('m', input_shape=input_shape, classifier_activation=None, first_strides=1)
        # base = EfficientNetV2M(input_shape=input_shape, classifier_activation=None, survivals=None)
        base.load_weights('checkpoints/efficientnetv2-m-21k-ft1k.h5', by_name=True)
        c5 = base.get_layer('add_50').output # 32x64
        c4 = base.get_layer('add_29').output # 128x256
        c3 = base.get_layer('add_10').output # 128x256
        base.summary()

        features = [c3, c4, c5]
        model_input = base.input
        model_output = fpn_model(features=features, fpn_times=4, activation='swish')
        model_output = classifier(model_output, num_classes=classes)

    else:
        raise print("Check your backbone name!")




    #
    # "deeplab v3+ aspp"
    # # x = _aspp(c5, 256)
    # x = layers.Dropout(rate=0.5)(x)
    # x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    # x = _conv_bn_relu(x, 48, 1, strides=1)
    #
    # x = Concatenate(out_size=aspp_size)([x, c2])
    # x = _conv_bn_relu(x, 256, 3, 1)
    # x = layers.Dropout(rate=0.5)(x)#
    # x = _conv_bn_relu(x, 256, 3, 1)
    # x = layers.Dropout(rate=0.1)(x)
    #
    # x = layers.Conv2D(20, 1, strides=1)(x)
    # x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

    return model_input, model_output

def classifier(x, num_classes=20):
    x = layers.Conv2D(num_classes, 1, strides=1, kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
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