import model
import utils
import keras
from table import generate_start_table
from generate_training_data import transposition, find_piece
from move_dict import VALID_MOVES, MOVE_TO_IDX
import numpy as np
import tensorflow as tf


def prepare_data(game, white_generation):
        """
        Processes each move of the game, updates the board state, and stores
        the (state, move) pairs for training.

    Args:
        game: The chess game to be processed.

    Returns:
        A tuple containing (list of board states, list of corresponding moves).
        """

        X = []
        y = []
        boardgame = generate_start_table()
        # Generate data
        for idx, move in enumerate(game):
            is_white = True if idx % 2 == 0 else False
            board_state = boardgame.copy()

            try:
                boardgame = transposition(boardgame, move, is_white)

                if white_generation == is_white:
                    move = move.replace('+', '').replace('#', '')

                    if move == 'O-O':
                        col, line = 8, 0
                    elif move == 'O-O-O':
                        col, line = 0, 8
                    else:
                        col, line = find_piece(board_state, move, is_white)

                    move = move.replace('x', '').replace('#', '').replace('+', '')
                    if move[0] == move[0].lower():
                        move = move[-2:]
                    elif move == 4:
                        move = move[0] + move[2:]

                    if move in VALID_MOVES:
                        X.append(board_state)
                        y.append([MOVE_TO_IDX[move], col, line])

            except Exception as e:
                return X, y

        return X, y


def choice_data(X, y, max_moves_per_game):
        """
        Randomly selects samples from the generated data and applies one-hot encoding to the labels.

        Args:
            X: List of board states.
            y: List of corresponding moves.

        Returns:
            Tuple containing (processed input data, labels in one-hot format).
        """

        sorted_idx = np.random.randint(low=0, high=len(X), size=max_moves_per_game)

        final_X, final_y = [], []

        for idx in sorted_idx:
            final_X.append(X[idx])
            final_y.append(y[idx])

        sparse_final_y = []
        for i, y in enumerate(final_y):
            move = [0 for _ in range(386)]
            move[y[0]] = 1
            col = [0 for _ in range(9)]
            col[y[1]] = 1
            line = [0 for _ in range(9)]
            line[y[2]] = 1

            sparse_final_y.append(tf.concat([move, col, line], axis=0))

        return np.array(final_X), np.array(sparse_final_y)


def fit_model(model, epochs, max_moves_per_game, limit_train_data_floor, limit_train_data_ceil, white_generation, test_model_on_training_data, limit_test_data_floor, limit_test_data_ceil):
    training_data = utils.clean_data(True)
    training_data = training_data[limit_train_data_floor:limit_train_data_ceil]

    for epoch in range(epochs):
        loss = 0
        for game in training_data:
            X, y = prepare_data(game, white_generation)
            X, y = choice_data(X, y, max_moves_per_game)
            x = model.train_on_batch(X, y)
            loss += x

        print(f"loss from epoch {epoch+1}: {loss/len(training_data)}")
        Xadrezia.save_weights('weights.weights.h5')
        if test_model_on_training_data:
            test_model(model, max_moves_per_game, limit_test_data_floor, limit_test_data_ceil, white_generation)


def test_model(model, max_moves_per_game, limit_train_data_floor, limit_train_data_ceil, white_generation):
    test_data = utils.clean_data(True)
    test_data = test_data[limit_train_data_floor:limit_train_data_ceil]
    loss = 0
    for game in test_data:
        X, y = prepare_data(game, white_generation)
        X, y = choice_data(X, y, max_moves_per_game)
        model.trainable = False
        pred = model.train_on_batch(X, y)
        model.trainable = True
        loss += pred

    print(f"test/validation loss: {loss / len(test_data)}")


if __name__ == '__main__':
    Xadrezia = model.Xadrezia()
    optimizer = keras.optimizers.Adam(learning_rate=15e-5)
    Xadrezia.compile(optimizer=optimizer, loss='categorical_crossentropy')
    fit_model(Xadrezia, epochs=10, max_moves_per_game=16, limit_train_data_floor=0, limit_train_data_ceil=300000, white_generation=True, test_model_on_training_data=True, limit_test_data_floor=300000, limit_test_data_ceil=400000)

