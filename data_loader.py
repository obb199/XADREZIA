import tensorflow as tf
import keras
import numpy as np
from generate_training_data import transposition, find_piece
from table import generate_start_table
from move_dict import MOVE_TO_IDX, VALID_MOVES


class DataGenerator(keras.utils.Sequence):
    def __init__(self, games,
                 batch_size=1,
                 moves_per_game=16,
                 shuffle=True,
                 for_white_generation=True):

        super().__init__()
        self.games = games
        self.moves_per_game = moves_per_game  # moves per game
        self.batch_size = batch_size  # games per batch
        self.shuffle = shuffle  # true or false to shuffle data after any epochs
        self.for_white_generation = for_white_generation
        self.on_epoch_end()  # call of the function
        self.indexes = np.arange(len(self.games))

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.games) / self.batch_size))

    def __choice_data(self, X, y):
        sorted_idx = np.random.randint(low=0, high=len(X), size=self.batch_size)

        final_X, final_y = [], []

        for idx in sorted_idx:
            if X[idx] not in final_X:
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

        return final_X, sparse_final_y

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [k for k in indexes]
        game = self.games[list_IDs_temp[0]]
        # Generate data
        X, y = self.__data_generation(game)
        X, y = self.__choice_data(X, y)

        return np.array(X), np.array(y)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.games))
        np.random.shuffle(self.indexes)

    def __data_generation(self, game):
        """Generates data containing batch_size samples"""
        X = []
        y = []
        boardgame = generate_start_table()
        # Generate data
        for idx, move in enumerate(game):
            is_white = True if idx % 2 == 0 else False
            board_state = boardgame.copy()

            try:
                boardgame = transposition(boardgame, move, is_white)

                if self.for_white_generation == is_white:
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
