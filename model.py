import tensorflow as tf
import keras
import numpy as np
import data_loader
import utils


def get_positional_encoding(model_dimension, batch_size):
    """
    Compute 2D positional encodings for a grid of given height and width.

    Args:
        model_dimension: The dimension of the model embeddings (must be even)
        height: Height of the grid (e.g., 8)
        width: Width of the grid (e.g., 8)
        batch_size: Number of samples in the batch

    Returns:
        A numpy array of shape (batch_size, height, width, model_dimension) with positional encodings
    """
    # Validate input
    if model_dimension % 2 != 0:
        raise ValueError("model_dimension must be even")

    # Initialize output array
    positional_encodings = np.zeros((batch_size, 8, 8, model_dimension))

    # Generate row and column indices for the grid
    row_indices = np.arange(8).reshape(1, 8, 1, 1)  # Shape: (1, height, 1, 1)
    col_indices = np.arange(8).reshape(1, 1, 8, 1)   # Shape: (1, 1, width, 1)

    # Pre-calculate denominators for all k values
    k_values = np.arange(model_dimension // 4)  # Half for row, half for col
    denominators = np.power(10000, 2 * k_values / model_dimension)  # Shape: (model_dimension // 4,)

    # Reshape denominators for broadcasting
    denominators = denominators.reshape(1, 1, 1, -1)  # Shape: (1, 1, 1, model_dimension // 4)

    # Calculate angles for row and column indices
    row_angle_rates = row_indices / denominators  # Shape: (1, height, 1, model_dimension // 4)
    col_angle_rates = col_indices / denominators  # Shape: (1, 1, width, model_dimension // 4)

    # Apply sine and cosine to row and column indices
    positional_encodings[..., 0:model_dimension//4] = np.sin(row_angle_rates)  # Row sin
    positional_encodings[..., model_dimension//4:model_dimension//2] = np.cos(row_angle_rates)  # Row cos
    positional_encodings[..., model_dimension//2:3*model_dimension//4] = np.sin(col_angle_rates)  # Col sin
    positional_encodings[..., 3*model_dimension//4:model_dimension] = np.cos(col_angle_rates)  # Col cos

    return positional_encodings


class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model, n_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.depth = d_model // n_heads

        self.wq = keras.layers.Dense(d_model)
        self.wk = keras.layers.Dense(d_model)
        self.wv = keras.layers.Dense(d_model)

        self.final_dense = keras.layers.Dense(d_model, activation='gelu')
        self.final_reshape = keras.layers.Reshape([-1, 8, 8, d_model])

    def split_heads(self, x, batch_size):
        """
        input shape: batch_size x seq_lenth x d_model
        intermediate shape: batch_size x seq_length/depth x num_heads x depth
        output shape: batch_size x num_heads x seq_length/depth x depth
        """
        x = tf.reshape(x, [batch_size, -1, self.n_heads, self.depth])
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return x

    def create_padding_mask(self, seq):
        # Identify positions where the token ID is 0 (padding)
        return tf.cast(tf.math.equal(seq, 0), tf.float32) * 1e-9

    def call(self, x):
        # linear combination with inputs
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # splitting heads for multihead attention
        batch_size = q.shape[0]
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # calculating attention
        scale = self.d_model ** 0.5
        qk_product = tf.matmul(q, k, transpose_b=True) / scale

        mask = self.create_padding_mask(qk_product)
        qk_product += mask

        qk_product = tf.nn.softmax(qk_product)

        attention_result = tf.matmul(qk_product, v)

        # return the values for the initial shape before splitting
        pre_output = tf.transpose(attention_result, perm=[0, 2, 1, 3])
        pre_output = tf.reshape(pre_output, [batch_size, -1, self.d_model])
        # linear combination with non linear activation
        output = self.final_dense(pre_output)
        output = tf.squeeze(self.final_reshape(output))

        if len(output.shape) == 3:
            self.final_reshape = tf.expand_dims(self.final_reshape, 0)
        return output


class Encoder(keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.norm_1 = keras.layers.LayerNormalization()
        self.norm_2 = keras.layers.LayerNormalization()
        self.ffn = keras.Sequential([keras.layers.Dense(d_model, activation='gelu'),
                                     keras.layers.Dropout(0.25),
                                     keras.layers.Dense(d_model, activation='gelu'),
                                     keras.layers.Dropout(0.25)])

    def call(self, x):
        auto_attention_result = self.mha.call(x)
        auto_attention_result = self.norm_1(auto_attention_result + x)

        encoder_result = self.ffn(auto_attention_result)
        encoder_result = self.norm_2(auto_attention_result + encoder_result)

        return encoder_result


class ResidualConvolution(keras.layers.Layer):
    def __init__(self, filters, prev_filters, kernel_size, **kwargs):
        super(ResidualConvolution, self).__init__(**kwargs)
        self.convs = keras.Sequential([keras.layers.Conv2D(filters, kernel_size=1, padding='same'),
                                       keras.layers.Conv2D(filters, kernel_size=kernel_size, padding='same'),
                                       keras.layers.BatchNormalization(),
                                       keras.layers.Activation('gelu'),
                                       keras.layers.Conv2D(filters, kernel_size=1, padding='same'),
                                       keras.layers.Conv2D(filters, kernel_size=kernel_size, padding='same'),
                                       keras.layers.BatchNormalization(),
                                       keras.layers.Activation('gelu')
                                       ])

        if prev_filters == filters:
            self.skip_connection = lambda x: x
        else:
            self.skip_connection = keras.layers.Conv2D(filters, kernel_size=1, padding='same')

    def call(self, x):
        x_skip = self.skip_connection(x)
        x = self.convs(x)

        return x + x_skip


class Xadrezia(keras.Model):
    def __init__(self, **kwargs):
        super(Xadrezia, self).__init__(**kwargs)
        self.convolutions = keras.Sequential([keras.layers.Conv2D(32, kernel_size=8, padding='same', strides=1),
                                              keras.layers.BatchNormalization(),
                                              keras.layers.Activation('gelu'),
                                              ResidualConvolution(64, 32, kernel_size=4),
                                              ResidualConvolution(128, 64, kernel_size=2),
                                              ResidualConvolution(256, 128, kernel_size=2),
                                              ])

        self.mha_encoders = keras.Sequential([Encoder(256, 8)]*4)

        self.move = keras.layers.Dense(386, activation='softmax')
        self.col = keras.layers.Dense(9, activation='softmax')
        self.row = keras.layers.Dense(9, activation='softmax')
        self.flatten = keras.layers.Flatten()

    def call(self, x):
        x = self.convolutions(x)  # Shape: (batch_size, 8, 8, d_model)
        pos_encoding = get_positional_encoding(256, 16)
        x = x + pos_encoding
        x = self.mha_encoders(x)
        move = self.move(self.flatten(x))  # Shape: (batch_size, 386)
        col = self.col(self.flatten(x))  # Shape: (batch_size, 9)
        row = self.row(self.flatten(x))  # Shape: (batch_size, 9)
        return tf.concat([move, col, row], 1)


if __name__ == '__main__':
    training_data = utils.clean_data(True)
    training_data = utils.shuffle(training_data, 1000000)
    length_data = len(training_data)

    Xadrezia = Xadrezia()
    optimizer = keras.optimizers.AdamW(learning_rate=1e-4)
    Xadrezia.compile(optimizer=optimizer, loss='categorical_crossentropy')
    gen_train = data_loader.DataGenerator(training_data[0:int(length_data*0.7)])
    gen_val = data_loader.DataGenerator(training_data[int(length_data*0.7):int(length_data*0.9)])
    gen_test = data_loader.DataGenerator(training_data[int(length_data*0.9):])
    Xadrezia.fit(gen_train, epochs=5, validation_data=gen_val)
    Xadrezia.save_weights('weights.weights.h5')
