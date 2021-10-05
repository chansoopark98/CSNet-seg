from tensorflow import keras
from model.model import csnet_seg_model

def seg_model_build(image_size, mode='cls', augment=False, weight_decay=0.0001, optimizer='sgd', num_classes=19):
    input, output = csnet_seg_model(backbone='efficientV2-s',
                                     input_shape=(image_size[0], image_size[1], 3),
                                     classes=num_classes, OS=16)
    model = keras.Model(input, output)
    return model
