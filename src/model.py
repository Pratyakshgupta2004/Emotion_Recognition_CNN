from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    SeparableConv2D,
    MaxPooling2D,
    Add,
    GlobalAveragePooling2D,
    Dropout,
    Dense,
    SpatialDropout2D
)

from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


L2 = l2(1e-4)


def residual_block(x, filters):

    residual = Conv2D(
        filters,
        (1,1),
        strides=(2,2),
        padding="same",
        use_bias=False,
        kernel_regularizer=L2
    )(x)

    residual = BatchNormalization()(residual)

    x = SeparableConv2D(
        filters,
        (3,3),
        padding="same",
        use_bias=False,
        depthwise_regularizer=L2,
        pointwise_regularizer=L2
    )(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = SeparableConv2D(
        filters,
        (3,3),
        padding="same",
        use_bias=False,
        depthwise_regularizer=L2,
        pointwise_regularizer=L2
    )(x)

    x = BatchNormalization()(x)

    x = SpatialDropout2D(0.15)(x)

    x = MaxPooling2D(
        pool_size=(3,3),
        strides=(2,2),
        padding="same"
    )(x)

    x = Add()([x, residual])

    return x


def build_model(input_shape=(48,48,1), num_classes=7):

    inputs = Input(shape=input_shape)

    x = Conv2D(
        32,
        (3,3),
        padding="same",
        use_bias=False,
        kernel_regularizer=L2
    )(inputs)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(
        32,
        (3,3),
        padding="same",
        use_bias=False,
        kernel_regularizer=L2
    )(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = residual_block(x,64)
    x = residual_block(x,128)
    x = residual_block(x,256)
    x = residual_block(x,512)

    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.5)(x)

    outputs = Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = Model(inputs, outputs)

    return model


if __name__ == "__main__":

    model = build_model()

    model.summary()