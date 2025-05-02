import tensorflow as tf
import keras


class XadreziaLoss(keras.losses.Loss):
    def __init__(self, name="XadreziaLoss"):
        super(XadreziaLoss, self).__init__(name=name)

    def call(self, y_true, y_pred):
        # y_true and y_pred are tuples/lists of 3 tensors
        true_move, true_col, true_row = y_true
        pred_move, pred_col, pred_row = y_pred

        # Compute categorical crossentropy for each output
        move_loss = keras.losses.categorical_crossentropy(true_move, pred_move) * 0.7
        col_loss = keras.losses.categorical_crossentropy(true_col, pred_col) * 0.15
        row_loss = keras.losses.categorical_crossentropy(true_row, pred_row) * 0.15

        # Sum the weighted losses and take the mean over the batch
        total_loss = move_loss + col_loss + row_loss
        return tf.reduce_mean(total_loss)
