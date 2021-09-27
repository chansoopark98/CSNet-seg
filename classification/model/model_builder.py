from classification.model.imageNet_model import ddrnet_23_slim
from classification.model.imageNet_model import seg_model
from classification.model.ddrnet_23_slim import ddrnet_23_slim_seg

def seg_model_build(image_size, mode='cls'):
    if mode =='cls':
        model = ddrnet_23_slim(input_shape=(image_size[0], image_size[1], 3))
    else:
        # model = seg_model(input_shape=(image_size[0], image_size[1], 3))
        model = ddrnet_23_slim_seg(input_shape=(image_size[0], image_size[1], 3), num_classes=20)

    return model
