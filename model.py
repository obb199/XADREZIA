import tensorflow as tf
import keras
import numpy as np


def get_positional_encoding(batch_size, height, width, d_model):
    """
    Generate a 2D positional encoding with shape (batch, height, width, d_model).

    For an odd d_model, the function computes sinusoidal encoding for (d_model - 1) channels
    (split equally between rows & columns) and appends an extra channel based on a combined row-col signal.

    Parameters:
        batch_size (int): Number of examples in the batch.
        height (int): Height of the 2D grid.
        width (int): Width of the 2D grid.
        d_model (int): Channel dimension of the encoding.

    Returns:
        pos_enc_batch (np.ndarray): A numpy array of shape (batch, height, width, d_model)
                                    containing the 2D positional encoding.
    """
    # If d_model is odd, work with d_model_even (which is d_model-1) and add one extra channel.
    if d_model % 2 == 0:
        d_model_even = d_model
        add_extra = False
    else:
        d_model_even = d_model - 1
        add_extra = True

    # We split the encoding channels equally between rows and columns.
    # d_model_even is assumed to be even.
    half_dim = d_model_even // 2

    # Create position indices for rows and columns.
    row_indices = np.arange(height)[:, np.newaxis]   # shape: (height, 1)
    col_indices = np.arange(width)[:, np.newaxis]      # shape: (width, 1)

    # Precompute frequencies for each channel index.
    # We use a simple formulation: frequency_j = 1 / 10000^(j/half_dim)
    j_indices = np.arange(half_dim)
    freqs = 1 / (10000 ** (j_indices / half_dim))  # shape: (half_dim,)

    # Initialize empty encodings for rows and columns.
    pos_row = np.zeros((height, half_dim))
    pos_col = np.zeros((width, half_dim))

    # Compute row encoding: for each channel use sin for even indices and cos for odd indices.
    for j in range(half_dim):
        # Multiply the row index with the frequency.
        if j % 2 == 0:
            pos_row[:, j] = np.sin(row_indices[:, 0] * freqs[j])
        else:
            pos_row[:, j] = np.cos(row_indices[:, 0] * freqs[j])

    # Compute column encoding similarly.
    for j in range(half_dim):
        if j % 2 == 0:
            pos_col[:, j] = np.sin(col_indices[:, 0] * freqs[j])
        else:
            pos_col[:, j] = np.cos(col_indices[:, 0] * freqs[j])

    # Expand dims so that row encoding is shaped (height, 1, half_dim) and
    # column encoding is shaped (1, width, half_dim).
    pos_row_expanded = pos_row[:, np.newaxis, :]        # (height, 1, half_dim)
    pos_col_expanded = pos_col[np.newaxis, :, :]          # (1, width, half_dim)

    # Broadcast to shape (height, width, half_dim) and then concatenate along the channel dimension.
    pos_row_broadcast = np.broadcast_to(pos_row_expanded, (height, width, half_dim))
    pos_col_broadcast = np.broadcast_to(pos_col_expanded, (height, width, half_dim))
    pos_enc = np.concatenate([pos_row_broadcast, pos_col_broadcast], axis=-1)  # shape: (height, width, d_model_even)

    # If the desired d_model is odd, append an extra channel.
    # For the extra channel, we will use a simple combined positional signal.
    if add_extra:
        # Generate a grid of row and column indices.
        grid_row = np.tile(np.arange(height)[:, np.newaxis], (1, width))
        grid_col = np.tile(np.arange(width)[np.newaxis, :], (height, 1))
        extra_channel = np.sin((grid_row + grid_col) / (height + width))
        extra_channel = extra_channel[..., np.newaxis]  # shape: (height, width, 1)
        pos_enc = np.concatenate([pos_enc, extra_channel], axis=-1)  # shape: (height, width, d_model)

    # Finally, add the batch dimension by tiling the same positional encoding for all items in the batch.
    pos_enc_batch = np.broadcast_to(pos_enc, (batch_size, height, width, d_model))
    return pos_enc_batch


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

        self.final_dense = keras.layers.Dense(d_model, activation='gelu')
        self.final_reshape = keras.layers.Reshape([height, width, d_model])  # Fixed shape assuming 8x8 grid

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

    def create_padding_mask(self, seq):
        # Identify positions where the token ID is 0 (padding)
        return tf.cast(tf.math.equal(seq, 0), tf.float32) * 1e-9

    def call(self, x):
        # Linear combination with inputs
        q = self.wq(x)  # Shape: (batch_size, 8, 8, d_model)
        k = self.wk(x)  # Shape: (batch_size, 8, 8, d_model)
        v = self.wv(x)  # Shape: (batch_size, 8, 8, d_model)

        # Splitting heads for multi-head attention
        q = self.split_heads(q)  # Shape: (batch_size, n_heads, 64, depth)
        k = self.split_heads(k)  # Shape: (batch_size, n_heads, 64, depth)
        v = self.split_heads(v)  # Shape: (batch_size, n_heads, 64, depth)

        # Calculating attention
        scale = tf.cast(self.d_model, tf.float32) ** 0.5
        qk_product = tf.matmul(q, k, transpose_b=True) / scale  # Shape: (batch_size, n_heads, 64, 64)

        mask = self.create_padding_mask(qk_product)
        qk_product += mask

        qk_product = tf.nn.softmax(qk_product)  # Shape: (batch_size, n_heads, 64, 64)

        attention_result = tf.matmul(qk_product, v)  # Shape: (batch_size, n_heads, 64, depth)

        # Return to the initial shape before splitting
        pre_output = tf.transpose(attention_result, perm=[0, 2, 1, 3])  # Shape: (batch_size, 64, n_heads, depth)
        batch_size = tf.shape(pre_output)[0]
        pre_output = tf.reshape(pre_output, [batch_size, -1, self.d_model])  # Shape: (batch_size, 64, d_model)

        # Linear combination with non-linear activation
        output = self.final_dense(pre_output)  # Shape: (batch_size, 64, d_model)
        output = self.final_reshape(output)  # Shape: (batch_size, 8, 8, d_model)

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
        self.convs = keras.Sequential([keras.layers.Conv2D(filters, kernel_size=1, padding='same'),
                                       keras.layers.Conv2D(filters, kernel_size=kernel_size, padding='same'),
                                       keras.layers.BatchNormalization(),
                                       keras.layers.Activation('gelu'),
                                       keras.layers.Conv2D(filters, kernel_size=1, padding='same'),
                                       keras.layers.Conv2D(filters, kernel_size=kernel_size, padding='same', strides=strides),
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
    def __init__(self, **kwargs):
        super(Xadrezia, self).__init__(**kwargs)
        self.convolutions = keras.Sequential([keras.layers.Conv2D(128, kernel_size=8, padding='same', strides=1),
                                              keras.layers.BatchNormalization(),
                                              keras.layers.Activation('gelu'),
                                              ResidualConvolution(256, 128, kernel_size=4, strides=1),
                                              ResidualConvolution(512, 256, kernel_size=2, strides=2),
                                              ])

        self.mha_encoders = keras.Sequential([Encoder(512, 8, 4, 4)]*2)

        self.move = keras.layers.Dense(386, activation='softmax')
        self.col = keras.layers.Dense(9, activation='softmax')
        self.row = keras.layers.Dense(9, activation='softmax')
        self.flatten = keras.layers.Flatten()

    def call(self, x):
        x = self.convolutions(x)  # Shape: (batch_size, 8, 8, d_model)
        pos_encoding = get_positional_encoding(batch_size=16, height=4, width=4, d_model=512)
        x = x + pos_encoding
        x = self.mha_encoders(x)
        move = self.move(self.flatten(x))  # Shape: (batch_size, 386)
        col = self.col(self.flatten(x))  # Shape: (batch_size, 9)
        row = self.row(self.flatten(x))  # Shape: (batch_size, 9)
        return tf.concat([move, col, row], 1)

