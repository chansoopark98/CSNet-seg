import efficientnet.keras as efn
import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import GlobalAveragePooling2D,  Reshape, Dense, multiply, Concatenate, \
    Conv2D, Add, Activation, Dropout ,BatchNormalization, DepthwiseConv2D, Lambda ,  UpSampling2D, SeparableConv2D, MaxPooling2D
from tensorflow.keras import backend as K
from functools import reduce

activation = tf.keras.activations.swish
# activation = tfa.activations.mish

MOMENTUM = 0.997
EPSILON = 1e-4

GET_EFFICIENT_NAME = {
    'B0': ['block3b_add', 'block5c_add', 'block7a_project_bn'],
    'B1': ['block3c_add', 'block5d_add', 'block7b_add'],
    'B2': ['block3c_add', 'block5d_add', 'block7b_add'],
    'B3': ['block3c_add', 'block5e_add', 'block7b_add'],
    'B4': ['block3d_add', 'block5f_add', 'block7b_add'],
    'B5': ['block3e_add', 'block5g_add', 'block7c_add'],
    'B6': ['block3f_add', 'block5h_add', 'block7c_add'],
    'B7': ['block3g_add', 'block5j_add', 'block7d_add'],
}

CONV_KERNEL_INITALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

w_bifpns = [64, 88, 112, 160, 224, 288, 384]
d_bifpns = [3, 4, 5, 6, 7, 7, 8]
image_sizes = [512, 640, 768, 896, 1024, 1280, 1408]

class weightAdd(keras.layers.Layer):
    def __init__(self, epsilon=1e-4, **kwargs):
        super(weightAdd, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        num_in = len(input_shape)
        self.w = self.add_weight(name=self.name,
                                 shape=(num_in,),
                                 initializer=keras.initializers.constant(1 / num_in),
                                 trainable=True,
                                 dtype=tf.float32)

    def call(self, inputs, **kwargs):
        w = keras.activations.relu(self.w)
        x = tf.reduce_sum([w[i] * inputs[i] for i in range(len(inputs))], axis=0)
        x = x / (tf.reduce_sum(w) + self.epsilon)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(weightAdd, self).get_config()
        config.update({
            'epsilon': self.epsilon
        })
        return config


def remove_dropout(model):
    for layer in model.layers:
        if isinstance(layer, Dropout):
            layer.rate = 0
    model_copy = keras.models.clone_model(model)
    model_copy.set_weights(model.get_weights())
    del model

    return model_copy



def create_efficientNet(base_model_name, pretrained=True, IMAGE_SIZE=[2048, 1024]):
    if pretrained is False:
        weights = None

    else:
        weights = "imagenet"

    if base_model_name == 'B0':
        base = efn.EfficientNetB0(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])

    elif base_model_name == 'B1':
        base = efn.EfficientNetB1(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])

    elif base_model_name == 'B2':
        base = efn.EfficientNetB2(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])

    elif base_model_name == 'B3':
        base = efn.EfficientNetB3(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])

    elif base_model_name == 'B4':
        base = efn.EfficientNetB4(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])

    elif base_model_name == 'B5':
        base = efn.EfficientNetB5(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])

    elif base_model_name == 'B6':
        base = efn.EfficientNetB6(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])

    elif base_model_name == 'B7':
        base = efn.EfficientNetB7(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])

    base = remove_dropout(base)
    base.trainable = True

    return base

def SeparableConvBlock(num_channels, kernel_size, strides, name, freeze_bn=False):
    f1 = SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                                use_bias=True, name=name+'/conv')
    f2 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=name+'/bn')
    # f2 = BatchNormalization(freeze=freeze_bn, name=f'{name}/bn')
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))



def csnet_extra_model(base_model_name, pretrained=True, IMAGE_SIZE=[2048, 1024]):
    base = create_efficientNet(base_model_name, pretrained, IMAGE_SIZE)

    layer_names = GET_EFFICIENT_NAME[base_model_name]
    # get extra layer
    #efficient_conv75 = base.get_layer('block2b_add').output  # 75 75 24
    # p3 = base.get_layer(layer_names[0]).output # 64 64 40
    # # p4 = base.get_layer('block4c_add').output
    # p5 = base.get_layer(layer_names[1]).output # 32 32 112
    # # p6 = base.get_layer('block6d_add').output
    p7 = base.get_layer(layer_names[2]).output # 16 16 320

    # features = p3, p5, p7
    # for i in range(3):
    #     print("times i : ", i+1)
    #     features = build_FPN(features, 64, times=i, normal_fusion=True)
    #
    p7 = SeparableConvBlock(num_channels=20, kernel_size=3, strides=1,
                       name='output_classifier')(p7)

    f_32 = UpSampling2D()(p7)
    f_64 = UpSampling2D()(f_32)
    f_128 = UpSampling2D()(f_64)
    f_256 = UpSampling2D()(f_128)
    f_512 = UpSampling2D()(f_256)

    f_512 = tf.math.argmax(f_512, axis=-1, output_type=tf.uint8)
    return base.input, f_512