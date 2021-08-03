from tensorflow.keras.layers import (
    MaxPooling2D, SeparableConv2D, UpSampling2D, Activation, Add, BatchNormalization, Conv2D)
from functools import reduce

MOMENTUM = 0.99
EPSILON = 0.0001

def ConvBlock(x, num_channels, kernel_size, strides, name, dilation_rate=1):
    x = Conv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                                use_bias=True, name=name+'/conv')(x)
    x = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=name+'/bn')(x)
    return x

def SeparableConvBlock(num_channels, kernel_size, strides, name, dilation_rate=1):
    f1 = SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                                use_bias=True, dilation_rate=dilation_rate, name=name+'/conv')
    f2 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=name+'/bn')
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))

def build_fpn(features, num_channels=64, activation='swish'):

    P3_in, P4_in, P5_in, P6_in, P7_in = features

    P7_U = UpSampling2D()(P7_in)

    P6_td = Add(name=f'fpn_cells/cell_{id}/fnode0/add')([P6_in, P7_U])
    P6_td = Activation(activation)(P6_td)
    P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                               name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)


    P6_U = UpSampling2D()(P6_td) # 4x4 to 9x9

    P5_td = Add(name=f'fpn_cells/cell_{id}/fnode1/add')([P5_in, P6_U]) # 9x9
    P5_td = Activation(activation)(P5_td)
    P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                               name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)

    P5_U = UpSampling2D()(P5_td) # 9x9 to 18x18
    P4_td = Add(name=f'fpn_cells/cell_{id}/fnode2/add')([P4_in, P5_U]) # 18x18
    P4_td = Activation(activation)(P4_td)
    P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                               name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')(P4_td)
    P4_U = UpSampling2D()(P4_td) # 18x18 to 36x36
    P3_out = Add(name=f'fpn_cells/cell_{id}/fnode3/add')([P3_in, P4_U])
    P3_out = Activation(activation)(P3_out)
    P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')(P3_out)
    P3_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out) # 36x36 to 18x18
    P4_out = Add(name=f'fpn_cells/cell_{id}/fnode4/add')([P4_in, P4_td, P3_D])
    P4_out = Activation(activation)(P4_out)
    P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')(P4_out)

    P4_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out) # 18x18 to 9x9
    P5_out = Add(name=f'fpn_cells/cell_{id}/fnode5/add')([P5_in, P5_td, P4_D])
    P5_out = Activation(activation)(P5_out)
    P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')(P5_out)

    # padding
    P5_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)  # 9x9 to 4x4

    P6_out = Add(name=f'fpn_cells/cell_{id}/fnode6/add')([P6_in, P6_td, P5_D])
    P6_out = Activation(activation)(P6_out)
    P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')(P6_out)

    P6_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
    P7_out = Add(name=f'fpn_cells/cell_{id}/fnode7/add')([P7_in, P6_D])
    P7_out = Activation(activation)(P7_out)
    P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')(P7_out)

    return [P3_out, P4_td, P5_td, P6_td, P7_out]