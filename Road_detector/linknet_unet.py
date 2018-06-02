from keras import layers
from keras.models import Model
from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Activation, Conv2DTranspose, concatenate, UpSampling2D
from keras.applications.vgg16 import VGG16
from resnet50_padding_same import ResNet50, identity_block
from resnet50_padding_same import conv_block as resnet_conv_block

bn_axis = 3


def linknet_residual_block(input_tensor, filters, shortcut=None):
    if shortcut is None:
        shortcut = input_tensor

    x = Conv2D(filters, (3, 3), padding='same')(input_tensor)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def linknet_conv_block(input_tensor, filters, res_blocks, stride=2):
    if stride == 1:
        x = input_tensor
    else:
        x = Conv2D(filters, (3, 3), strides=(stride, stride), padding='same')(input_tensor)
        x = BatchNormalization(axis=bn_axis)(x)

    for i in range(res_blocks):
        x = linknet_residual_block(x, filters)

    return x

def linknet_deconv_block(input_tensor, filters_in, filters_out):
    x = Conv2D(int(filters_in/4), (1, 1), padding='same')(input_tensor)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(int(filters_in/4), (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters_out, (1, 1), padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    return x

def get_linknet(input_shape=(512, 512)):
    inp = Input(input_shape + (9,))

    x = Conv2D(64, (7, 7), padding='same', strides=(2, 2))(inp) #, strides=(2, 2)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
#    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    enc1 = linknet_conv_block(x, 64, 3, stride=1) #, stride=1
    enc2 = linknet_conv_block(enc1, 128, 4)
    enc3 = linknet_conv_block(enc2, 256, 6)
    enc4 = linknet_conv_block(enc3, 384, 3)
    enc5 = linknet_conv_block(enc4, 512, 3)

    dec5 = linknet_deconv_block(enc5, 512, 384)
    dec5 = layers.add([dec5, enc4])
    dec4 = linknet_deconv_block(dec5, 384, 256)
    dec4 = layers.add([dec4, enc3])
    dec3 = linknet_deconv_block(dec4, 256, 128)
    dec3 = layers.add([dec3, enc2])
    dec2 = linknet_deconv_block(dec3, 128, 64)
    dec2 = layers.add([dec2, enc1])
    dec1 = linknet_deconv_block(dec2, 64, 64)

    x = Conv2D(48, (3, 3), padding='same')(dec1)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = Model(inp, x)

    return model

def get_resnet50_linknet(input_shape, channel_no, weights='imagenet'):
    inp = Input(input_shape + (channel_no,))

    x = Conv2D(
        64, (7, 7), strides=(2, 2), padding='same', name='conv1')(inp)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = resnet_conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    enc1 = x

    x = resnet_conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    enc2 = x

    x = resnet_conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    enc3 = x

    x = resnet_conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    enc4 = x

    dec4 = linknet_deconv_block(enc4, 2048, 1024)
    dec4 = layers.add([dec4, enc3])
    dec3 = linknet_deconv_block(dec4, 1024, 512)
    dec3 = layers.add([dec3, enc2])
    dec2 = linknet_deconv_block(dec3, 512, 256)
    dec2 = layers.add([dec2, enc1])
    dec1 = linknet_deconv_block(dec2, 256, 64)

    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(dec1)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(48, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = Model(inp, x)
    if weights == 'imagenet':
        resnet = ResNet50(input_shape=input_shape + (3,), include_top=False, weights=weights)
        for i in range(2, len(resnet.layers)-1):
            model.layers[i].set_weights(resnet.layers[i].get_weights())
            model.layers[i].trainable = False

    return model
