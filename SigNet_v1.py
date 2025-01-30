from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda, Flatten, Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.saving import register_keras_serializable

# ✅ Register the Euclidean distance as a global function
@register_keras_serializable()
def euclidean_distance(vectors):
    """
    Computes the Euclidean distance between two feature vectors.
    """
    x, y = vectors
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

# ✅ Define Contrastive Loss
@register_keras_serializable()
def contrastive_loss(y_true, y_pred):
    """
    Computes the Contrastive Loss as defined in Hadsell-et-al. (2006).
    """
    margin = 1.0
    y_true = K.cast(y_true, y_pred.dtype)
    positive_loss = (1 - y_true) * K.square(y_pred)
    negative_loss = y_true * K.square(K.maximum(margin - y_pred, 0))
    return K.mean(positive_loss + negative_loss)

# ✅ Define the SigNet Base Network Architecture
def create_base_network_signet(input_shape):
    """
    Builds the base convolutional neural network used in SigNet.
    """
    seq = Sequential()

    # First convolutional block
    seq.add(Conv2D(96, (11, 11), activation='relu', strides=(4, 4), 
                   input_shape=input_shape, kernel_initializer='glorot_uniform'))
    seq.add(BatchNormalization())
    seq.add(MaxPooling2D((3, 3), strides=(2, 2)))
    seq.add(ZeroPadding2D((2, 2)))

    # Second convolutional block
    seq.add(Conv2D(256, (5, 5), activation='relu', kernel_initializer='glorot_uniform'))
    seq.add(BatchNormalization())
    seq.add(MaxPooling2D((3, 3), strides=(2, 2)))
    seq.add(Dropout(0.3))
    seq.add(ZeroPadding2D((1, 1)))

    # Third convolutional block
    seq.add(Conv2D(384, (3, 3), activation='relu', kernel_initializer='glorot_uniform'))
    seq.add(ZeroPadding2D((1, 1)))

    # Fourth convolutional block
    seq.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='glorot_uniform'))
    seq.add(MaxPooling2D((3, 3), strides=(2, 2)))
    seq.add(Dropout(0.3))

    # Fully connected layers
    seq.add(Flatten())
    seq.add(Dense(1024, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.0005)))
    seq.add(Dropout(0.5))
    seq.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.0005)))

    return seq

# ✅ Define the Siamese Network with Contrastive Loss
def create_siamese_network(input_shape):
    """
    Builds the Siamese Neural Network for signature verification.
    """
    input_a = Input(shape=input_shape, name="input_a")
    input_b = Input(shape=input_shape, name="input_b")

    # Use the same CNN encoder for both inputs
    base_network = create_base_network_signet(input_shape)

    encoded_a = base_network(input_a)
    encoded_b = base_network(input_b)

    # Compute Euclidean distance
    distance = Lambda(euclidean_distance)([encoded_a, encoded_b])

    model = Model(inputs=[input_a, input_b], outputs=distance)
    return model
