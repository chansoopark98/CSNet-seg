# from efficientnet_v2 import *
from model.efficientnet_v2 import *
# from model.efficientnet_v2 import EfficientNetV2S
from model.resnet101 import *
from tensorflow.keras import layers
from model.fpn_model import deepLabV3Plus, SepConv_BN, DECAY, proposed, proposed_experiments

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
    for EfficientNetV2S
    32x64 = 'add_34'
    64x128 = 'add_7'
    128x256 = 'add_4'
    """
    features = [c2, c5]

    model_input = base.input
    # model_output, aspp_aux = deepLabV3Plus(features=features, fpn_times=2, activation='swish', mode='deeplabv3+')
    # model_output, aspp_aux, skip_aux = proposed(features=features, fpn_times=2, activation='relu', mode='deeplabv3+')
    decoder_output, edge_output, aspp_output = proposed_experiments(features=features, activation='relu')
    """
    model_output: 128x256
    aspp_aux: 64x128
    dec_aux: 128x256"""

    total_cls = classifier(decoder_output, num_classes=classes, upper=4, name='output')
    edge_cls = edge_classifier(edge_output, upper=4, name='edge')
    # body_cls = classifier(body_output, num_classes=classes, upper=4, name='body')
    aspp_aux_output = classifier(aspp_output, num_classes=classes, upper=4, name='aspp')
    eff_aux_output = classifier(c3, num_classes=classes, upper=8, name='eff')

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

    model_output = [total_cls, edge_cls, aspp_aux_output, eff_aux_output]

    return model_input, model_output


def classifier(x, num_classes=19, upper=4, name=None):
    x = layers.Conv2D(num_classes, 1, strides=1,
                      kernel_regularizer=DECAY,
                      kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = layers.UpSampling2D(size=(upper, upper), interpolation='bilinear', name=name)(x)
    return x

def edge_classifier(x, upper=2, name=None):
    x = layers.UpSampling2D(size=(upper, upper), interpolation='bilinear', name=name)(x)
    return x