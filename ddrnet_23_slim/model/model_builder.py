from ddrnet_23_slim.model.imageNet_model import DDRNet
import tensorflow as tf

def seg_model_build(image_size, mode='cls', augment=False, weight_decay=0.0001, optimizer='sgd', num_classes=19):
    if mode == 'cls':
        base = DDRNet(augment=augment, weight_decay=weight_decay, sync_batch=False,
                      optimizer=optimizer, bn_type='default')
        model = base.classification_model(input_shape=(image_size[0], image_size[1], 3))

    else:
        base = DDRNet(augment=augment, weight_decay=weight_decay, sync_batch=True,
                      optimizer=optimizer, bn_type='sync')
        model = base.seg_model(input_shape=(image_size[0], image_size[1], 3), num_classes=num_classes, augment=augment)

    # set weight initializers
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel_initializer = tf.keras.initializers.he_normal()
        if hasattr(layer, 'depthwise_initializer'):
            layer.depthwise_initializer = tf.keras.initializers.he_normal()

    return model
