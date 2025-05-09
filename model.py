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
    if d_model % 2 != 0:
        raise "d_model parameter is not divisible by two."

    half_dim = d_model // 2

    # Generate frequencies using original Transformer's formula
    j = np.arange(half_dim)
    freqs = 1 / (10000 ** (2 * j / d_model))

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

    pos_enc = np.zeros((height, width, d_model))
    pos_enc[..., :half_dim] = pos_row + pos_col  # Combine row/col features
    pos_enc[..., half_dim:] = pos_row * pos_col  # Add multiplicative interactions

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
            keras.layers.Dense(d_model, activation='relu'),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(d_model, activation='relu'),
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

    def build(self, input_shape):
        # Build sub-layers
        self.wq.build(input_shape)
        self.wk.build(input_shape)
        self.wv.build(input_shape)
        self.final_dense.build([None, input_shape[1] * input_shape[2], self.d_model])
        super(MultiHeadAttention, self).build(input_shape)

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
        output = self.final_dense(pre_output)
        output = self.final_reshape(output)  # Shape: (batch_size, height, width, d_model)

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

        self.final_dense = keras.Sequential([keras.layers.Dense(d_model, activation='relu'),
                                             keras.layers.LayerNormalization(),
                                             keras.layers.Dropout(0.25),
                                             keras.layers.Dense(d_model, activation='relu'),
                                             keras.layers.LayerNormalization(),
                                             keras.layers.Dropout(0.25)
                                             ])

    def build(self, input_shape):
        # Ensure all sub-layers are built
        self.mha.build(input_shape)
        self.norm_1.build(input_shape)
        self.final_dense.build(input_shape)
        self.norm_2.build(input_shape)
        super(Encoder, self).build(input_shape)

    def call(self, x):
        auto_attention_result = self.mha.call(x)
        auto_attention_result = self.norm_1(auto_attention_result + x)

        encoder_result = self.final_dense(auto_attention_result)
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

    def __init__(self, filters, **kwargs):
        super(ResidualConvolution, self).__init__(**kwargs)
        self.conv_1 = keras.Sequential(
            [keras.layers.Conv2D(filters, kernel_size=(1, 1), padding='same', use_bias=False),
             keras.layers.Activation('relu'),
             keras.layers.BatchNormalization()])

        self.conv_2 = keras.Sequential(
            [keras.layers.Conv2D(filters, kernel_size=(2, 2), padding='same', use_bias=False),
             keras.layers.Activation('relu'),
             keras.layers.BatchNormalization()])

        self.conv_3 = keras.Sequential(
            [keras.layers.Conv2D(filters, kernel_size=(1, 5), padding='same', use_bias=False),
             keras.layers.Conv2D(filters, kernel_size=(5, 1), padding='same', use_bias=False),
             keras.layers.Activation('relu'),
             keras.layers.BatchNormalization()])

        self.conv_4 = keras.Sequential(
            [keras.layers.Conv2D(filters, kernel_size=(8, 1), padding='same', use_bias=False),
             keras.layers.Conv2D(filters, kernel_size=(1, 8), padding='same', use_bias=False),
             keras.layers.Activation('relu'),
             keras.layers.BatchNormalization()])

        self.se_block = keras.Sequential([keras.layers.GlobalAvgPool2D(),
                                          keras.layers.Dense(filters * 4, activation='relu'),
                                          keras.layers.BatchNormalization(),
                                          keras.layers.Dense(filters, activation='relu'),
                                          keras.layers.BatchNormalization(),
                                          keras.layers.Dense(filters * 4, activation='sigmoid'),
                                          keras.layers.Reshape([1, 1, filters * 4])])

        self.skip_connection = keras.layers.Conv2D(filters * 4, kernel_size=1, padding='same', use_bias=False)

    def call(self, x):
        x_skip = self.skip_connection(x)
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        x = tf.concat([x1, x2, x3, x4], -1)
        x = self.se_block(x) * x

        return x + x_skip


class WeightedAverage(keras.layers.Layer):
    def __init__(self, base_params, **kwargs):
        super(WeightedAverage, self).__init__(**kwargs)
        # Use a smaller weight tensor, e.g., per channel
        self.weights_for_avg = tf.Variable(tf.random.normal([1, 1, base_params]), trainable=True)
        self.avg = keras.layers.GlobalAvgPool2D()

    def call(self, x):
        return self.avg(x * self.weights_for_avg)


class Xadrezia(keras.Model):
    """
        Main model for chess move analysis and prediction.

        Architecture:
            - Inception Squeeze-and-Excitation Residual convolutional blocks
            - Encoders with self-attention
            - Heads for move, column, and row prediction

        Methods:
            call(x): Processes the input and returns concatenated predictions.
"""

    def __init__(self, height=8, width=8, base_params=256, encoders_per_group=32, **kwargs):
        super(Xadrezia, self).__init__(**kwargs)
        self.base_params = base_params
        self.pos_encoding = get_positional_encoding(height, width, base_params)
        self.convolutions = keras.Sequential(
            [keras.layers.Conv2D(base_params // 4, kernel_size=5, padding='same', strides=1, use_bias=False),
             keras.layers.BatchNormalization(),
             keras.layers.Activation('relu'),
             ResidualConvolution(base_params // 4),
             ])

        self.encoders = [Encoder(base_params, 8, 8, 8)] * encoders_per_group

        self.weighted_average = WeightedAverage(base_params)

        self.move = keras.Sequential([keras.layers.Dense(4098, activation='softmax')])

    def call(self, x):
        x = self.convolutions(x)
        x = x + self.pos_encoding
        prev_result = x
        for encoder in self.encoders:
            x = encoder(x) + prev_result
            prev_result = x
        x = self.weighted_average(x)
        move_probabilities = self.move(x)  # Shape: (batch_size, 4098)
        return move_probabilities

"""
            ┌────────────────────────────┐
            │        Input (8x8x7)       │
            └────────────┬───────────────┘
                         │
                         ▼
            ┌────────────────────────────┐
            │   Convolutional Stack      │
            │----------------------------│
            │   Conv2D + BN + GELU       │ ◄── Feature extraction
            │   SE-ResidualConv x 6      │
            │   (progressively deepens)  │
            └────────────┬───────────────┘
                         │
                         ▼
            ┌────────────────────────────┐
            │Multi-Head Specialists [xN]:│
            │----------------------------│
            │ - SE-ResidualConv (d_model)│◄── Feature extraction
            │ - PosEnc fusion            │
            │ - Transformer Encoders x6  │
            └────────────┬───────────────┘
                         │
                         ▼
            ┌────────────────────────────┐
            │     Concatenate Features   │ ◄── Combines all specialists results
            └────────────┬───────────────┘
                         │
                         ▼
            ┌────────────────────────────┐
            │          Flatten           │
            └────────────┬───────────────┘
                         │
                         ▼
            ┌────────────────────────────┐
            │         Dense Block        │
            │            With            │ ◄── Output: move probabilities
            │      Softmax Activation    │ 
            └────────────────────────────┘
"""
