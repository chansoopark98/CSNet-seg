from tensorflow import keras
from model.model import csnet_seg_model, build_generator, build_discriminator

def seg_model_build(image_size, mode='cls', augment=False, weight_decay=0.0001, optimizer='sgd', num_classes=19):
    input, output = csnet_seg_model(backbone='efficientV2-s',
                                     input_shape=(image_size[0], image_size[1], 3),
                                     classes=num_classes, OS=16)
    model = keras.Model(input, output)

    return model

def build_gen(image_size, num_classes=19):
    model_input, model_output = build_generator(input_shape=image_size, classes=num_classes)

    return model_input, model_output

def build_dis(image_size):
    model_input, model_output = build_discriminator(image_size=image_size)

    return model_input, model_output