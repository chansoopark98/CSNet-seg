import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow_addons as tfa

def conv3x3(out_planes, stride=1):
    return layers.Conv2D(kernel_size=(3,3), filters=out_planes, strides=stride, padding="same",
                       use_bias=False)

"""
Creates a residual block with two 3*3 conv's
in paper it's represented by RB block
"""
basicblock_expansion = 1
def basic_block(x_in, planes, stride=1, downsample=None, no_relu=False):
    residual = x_in

    x = conv3x3(planes, stride)(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = conv3x3(planes,)(x)
    x = layers.BatchNormalization()(x)

    if downsample is not None:
        residual = downsample

    # x += residual
    x = layers.Add()([x, residual])

    if not no_relu:
        x = layers.Activation("relu")(x)

    return x

"""
creates a bottleneck block of 1*1 -> 3*3 -> 1*1
"""
bottleneck_expansion = 2
def bottleneck_block(x_in, planes, stride=1, downsample=None, no_relu=True):
    residual = x_in

    x = layers.Conv2D(filters=planes, kernel_size=(1,1), use_bias=False)(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters=planes, kernel_size=(3,3), strides=stride, padding="same",use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters=planes* bottleneck_expansion, kernel_size=(1,1), use_bias=False)(x)
    x= layers.BatchNormalization()(x)

    if downsample is not None:
        residual = downsample

    # x += residual
    x = layers.Add()([x, residual])

    if not no_relu:
        x = layers.Activation("relu")(x)

    return  x

# Deep Aggregation Pyramid Pooling Module
def DAPPPM(x_in, branch_planes, outplanes):
    input_shape = tf.keras.backend.int_shape(x_in)
    height = input_shape[1]
    width = input_shape[2]
    # Average pooling kernel size
    kernal_sizes_height = [5, 9, 17, height]
    kernal_sizes_width =  [5, 9, 17, width]
    # Average pooling strides size
    stride_sizes_height = [2, 4, 8, height]
    stride_sizes_width =  [2, 4, 8, width]
    x_list = []

    # y1
    scale0 = layers.BatchNormalization()(x_in)
    scale0 = layers.Activation("relu")(scale0)
    scale0 = layers.Conv2D(branch_planes, kernel_size=(1,1), use_bias=False, )(scale0)
    x_list.append(scale0)

    for i in range( len(kernal_sizes_height)):
        # first apply average pooling
        temp = layers.AveragePooling2D(pool_size=(kernal_sizes_height[i],kernal_sizes_width[i]),
                                       strides=(stride_sizes_height[i],stride_sizes_width[i]),
                                       padding="same")(x_in)
        temp = layers.BatchNormalization()(temp)
        temp = layers.Activation("relu")(temp)
        # then apply 1*1 conv
        temp = layers.Conv2D(branch_planes, kernel_size=(1, 1), use_bias=False, )(temp)
        # then resize using bilinear
        temp = tf.image.resize(temp, size=(height,width), )
        # add current and previous layer output
        temp = layers.Add()([temp, x_list[i]])
        temp = layers.BatchNormalization()(temp)
        temp = layers.Activation("relu")(temp)
        # at the end apply 3*3 conv
        temp = layers.Conv2D(branch_planes, kernel_size=(3, 3), use_bias=False, padding="same")(temp)
        # y[i+1]
        x_list.append(temp)

    # concatenate all
    combined = layers.concatenate(x_list, axis=-1)

    combined = layers.BatchNormalization()(combined)
    combined = layers.Activation("relu")(combined)
    combined = layers.Conv2D(outplanes, kernel_size=(1, 1), use_bias=False, )(combined)

    shortcut = layers.BatchNormalization()(x_in)
    shortcut = layers.Activation("relu")(shortcut)
    shortcut = layers.Conv2D(outplanes, kernel_size=(1, 1), use_bias=False, )(shortcut)

    # final = combined + shortcut
    final = layers.Add()([combined, shortcut])

    return final

"""
Segmentation head 
3*3 -> 1*1 -> rescale
"""
def segmentation_head(x_in, interplanes, outplanes, scale_factor=None):
    x = layers.BatchNormalization()(x_in)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(interplanes, kernel_size=(3, 3), use_bias=False, padding="same")(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(outplanes, kernel_size=(1, 1), use_bias=range, padding="valid")(x)

    if scale_factor is not None:
        input_shape = tf.keras.backend.int_shape(x)
        height2 = input_shape[1] * scale_factor
        width2 = input_shape[2] * scale_factor
        x = tf.image.resize(x, size =(height2, width2))

    return x

"""
apply multiple RB or RBB blocks.
x_in: input tensor
block: block to apply it can be RB or RBB
inplanes: input tensor channes
planes: output tensor channels
blocks_num: number of time block to applied
stride: stride
expansion: expand last dimension
"""
def make_layer(x_in, block, inplanes, planes, blocks_num, stride=1, expansion=1):
    downsample = None
    if stride != 1 or inplanes != planes * expansion:
        downsample = layers.Conv2D(((planes * expansion)), kernel_size=(1, 1),strides=stride, use_bias=False)(x_in)
        downsample = layers.BatchNormalization()(downsample)
        downsample = layers.Activation("relu")(downsample)

    x = block(x_in, planes, stride, downsample)
    for i in range(1, blocks_num):
        if i == (blocks_num - 1):
            x = block(x, planes, stride=1, no_relu=True)
        else:
            x = block(x, planes, stride=1, no_relu=False)

    return x


