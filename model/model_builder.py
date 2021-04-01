import tensorflow as tf
from tensorflow import keras
from model.model import csnet_extra_model

# train.py에서 priors를 변경하면 여기도 수정해야함
def model_build(base_model_name, pretrained=True, image_size=[512, 512]):

    inputs, output = csnet_extra_model(base_model_name, pretrained, image_size)
    model = keras.Model(inputs, outputs=output)
    return model

