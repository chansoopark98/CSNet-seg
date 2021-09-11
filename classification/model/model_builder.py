from imageNet_model import ddrnet_23_slim

def seg_model_build(image_size):
    model = ddrnet_23_slim(input_shape=(image_size[0], image_size[1], 3))

    return model
