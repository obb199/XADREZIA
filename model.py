import tensorflow as tf
import keras
import numpy as np


def get_positional_encoding(height, width, d_model):
    """
    Generates 2D positional encoding with vectorized operations.

    Args:
        height: Height of the grid
        width: Width of the grid
        d_model: Dimension of the encoding

    Returns:
        Tensor of shape (height, width, d_model)
    """
    if d_model % 2 == 0:
        d_model_even = d_model
        add_extra = False
    else:
        d_model_even = d_model - 1
        add_extra = True

    half_dim = d_model_even // 2

    # Generate frequencies using original Transformer's formula
    j = np.arange(half_dim)
    freqs = 1 / (10000 ** (2 * j / d_model_even))

    # Row encoding (vectorized)
    row_indices = np.arange(height)[:, np.newaxis]  # (height, 1)
    angles_row = row_indices * freqs  # (height, half_dim)
    pos_row = np.zeros((height, half_dim))
    pos_row[:, 0::2] = np.sin(angles_row[:, 0::2])
    pos_row[:, 1::2] = np.cos(angles_row[:, 1::2])

    # Column encoding (vectorized)
    col_indices = np.arange(width)[:, np.newaxis]  # (width, 1)
    angles_col = col_indices * freqs  # (width, half_dim)
    pos_col = np.zeros((width, half_dim))
    pos_col[:, 0::2] = np.sin(angles_col[:, 0::2])
    pos_col[:, 1::2] = np.cos(angles_col[:, 1::2])

    # Broadcast and concatenate
    pos_row = pos_row[:, np.newaxis, :]  # (height, 1, half_dim)
    pos_col = pos_col[np.newaxis, :, :]  # (1, width, half_dim)

    pos_enc = np.zeros((height, width, d_model_even))
    pos_enc[..., :half_dim] = pos_row + pos_col  # Combine row/col features
    pos_enc[..., half_dim:] = pos_row * pos_col  # Add multiplicative interactions

    # Handle odd dimensionality
    if add_extra:
        # Use geometric mean frequency for the extra channel
        geom_freq = 1 / (10000 ** ((2 * half_dim) / (d_model_even + 1)))
        row_geom = np.sin(row_indices * geom_freq)
        col_geom = np.cos(col_indices * geom_freq)
        extra_channel = row_geom[:, np.newaxis] * col_geom[np.newaxis, :]
        pos_enc = np.concatenate([pos_enc, extra_channel[:, :, np.newaxis]], axis=-1)

    return tf.constant(pos_enc, dtype=tf.float32)


class MultiHeadAttention(keras.layers.Layer):
    """
    Multi-head attention implementation for chess position analysis.

    Args:
        d_model: Model dimensionality.
        n_heads: Number of attention heads.
        **kwargs: Additional arguments for the Layer class.
"""

    def __init__(self, d_model, n_heads, height, width, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.depth = d_model // n_heads

        self.wq = keras.layers.Dense(d_model)
        self.wk = keras.layers.Dense(d_model)
        self.wv = keras.layers.Dense(d_model)

        self.final_dense = keras.Sequential([
            keras.layers.Dense(d_model, activation='gelu'),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(d_model, activation='gelu'),
            keras.layers.Dropout(0.25)
        ])

        self.final_reshape = keras.layers.Reshape([height, width, d_model])  # Fixed shape assuming height x width grid

    def split_heads(self, x):
        """
        Split the last dimension into (n_heads, depth) and transpose.

        Input shape: (batch_size, 8, 8, d_model)
        Intermediate shape: (batch_size, 64, n_heads, depth)
        Output shape: (batch_size, n_heads, 64, depth)
        """
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, -1, self.n_heads, self.depth])
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return x

    def call(self, x):
        # Linear combination with inputs
        q = self.wq(x)  # Shape: (batch_size, height, width, d_model)
        k = self.wk(x)  # Shape: (batch_size, height, width, d_model)
        v = self.wv(x)  # Shape: (batch_size, height, width, d_model)

        # Splitting heads for multi-head attention
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Calculating attention
        scale = tf.cast(self.d_model, tf.float32) ** 0.5
        qk_product = tf.matmul(q, k, transpose_b=True) / scale
        qk_product = tf.nn.softmax(qk_product)
        attention_result = tf.matmul(qk_product, v)

        # Return to the initial shape before splitting
        pre_output = tf.transpose(attention_result, perm=[0, 2, 1, 3])
        batch_size = tf.shape(pre_output)[0]
        pre_output = tf.reshape(pre_output, [batch_size, -1, self.d_model])

        # Linear combination with non-linear activation
        output = self.final_dense(pre_output)  # Shape: (batch_size, height * width, d_model)
        output = self.final_reshape(output)  # Shape: (batch_size, height, width, d_model)
        # output = self.final_conv(output)

        return output


class Encoder(keras.layers.Layer):
    """
        Encoder layer with self-attention and feed-forward network.

        Args:
            d_model: Model dimensionality.
            num_heads: Number of attention heads.
            **kwargs: Additional arguments for the Layer class.
"""

    def __init__(self, d_model, num_heads, height, width, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.mha = MultiHeadAttention(d_model, num_heads, height, width)
        self.norm_1 = keras.layers.LayerNormalization()
        self.norm_2 = keras.layers.LayerNormalization()
        self.ffn = keras.Sequential([keras.layers.Dense(d_model, activation='gelu'),
                                     keras.layers.LayerNormalization(),
                                     keras.layers.Dropout(0.25)])

    def call(self, x):
        auto_attention_result = self.mha.call(x)
        auto_attention_result = self.norm_1(auto_attention_result + x)

        encoder_result = self.ffn(auto_attention_result)
        encoder_result = self.norm_2(auto_attention_result + encoder_result)

        return encoder_result


class ResidualConvolution(keras.layers.Layer):
    """
        Bloco convolucional residual com conexão skip.

        Args:
            filters: Número de filtros para as camadas convolucionais.
            prev_filters: Número de filtros da camada anterior.
            kernel_size: Tamanho do kernel para convolução.
            **kwargs: Argumentos adicionais para a classe Layer.
        """

    def __init__(self, filters, prev_filters, kernel_size, strides, **kwargs):
        super(ResidualConvolution, self).__init__(**kwargs)
        self.convs = keras.Sequential(
            [keras.layers.Conv2D(filters, kernel_size=kernel_size, padding='same', strides=strides),
             keras.layers.BatchNormalization(),
             keras.layers.Activation('gelu')
             ])

        if prev_filters != filters or strides >= 2:
            self.skip_connection = keras.layers.Conv2D(filters, kernel_size=1, padding='same', strides=strides)
        else:
            self.skip_connection = lambda x: x

    def call(self, x):
        x_skip = self.skip_connection(x)
        x = self.convs(x)

        return x + x_skip


class Xadrezia(keras.Model):
    """
        Main model for chess move analysis and prediction.

        Architecture:
            - Residual convolutional blocks
            - Encoders with self-attention
            - Heads for move, column, and row prediction

        Methods:
            call(x): Processes the input and returns concatenated predictions.
"""

    def __init__(self, height=8, width=8, d_model=256, **kwargs):
        super(Xadrezia, self).__init__(**kwargs)
        self.d_model = d_model
        self.pos_encoding = get_positional_encoding(height, width, d_model)
        self.convolutions = keras.Sequential(
            [keras.layers.Conv2D(128, kernel_size=4, padding='same', strides=1),
             keras.layers.Activation('gelu'),
             ])

        self.pre_mha_conv = ResidualConvolution(d_model, 128, kernel_size=2, strides=1)
        self.mha_encoders = keras.Sequential([Encoder(d_model, 8, 8, 8)] * 8)

        self.move = keras.Sequential([keras.layers.Dense(d_model, activation='gelu'),
                                      keras.layers.LayerNormalization(),
                                      keras.layers.Dropout(0.25),
                                      keras.layers.Dense(4098, activation='softmax')])

        #self.maxpool = keras.layers.GlobalAvgPool2D()
        self.flatt = keras.layers.Flatten()

    def call(self, x):
        x = self.convolutions(x)  # Shape: (batch_size, 8, 8, d_model)
        x = self.pre_mha_conv(x) + self.pos_encoding * 0.25
        x = self.mha_encoders(x)
        move_probabilities = self.move(self.flatt(x))  # Shape: (batch_size, 4098)
        return move_probabilities
