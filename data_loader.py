import keras
import numpy as np
from generate_training_data import transposition, find_piece
from table import generate_start_table
from moves import MOVE_TO_IDX, VALID_MOVES
from convertions import COLS_CONVERTION


class DataGenerator(keras.utils.Sequence):
    """
        Data generator for training chess models.

        This class implements a data generator that processes chess games,
        generates board states and corresponding moves, and provides batches
        for training machine learning models.

        Args:
            games: List of chess games to be processed.
            batch_size: Number of games per batch (default: 1).
            moves_per_game: Maximum number of moves to be considered per game (default: 16).
            shuffle: Boolean indicating whether to shuffle the data after each epoch (default: True).
            for_white_generation: Boolean indicating whether to generate data for white moves (default: True).
"""
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
        """
       Calculates the number of batches per epoch.

       Returns:
           Integer representing the number of batches per epoch.
"""
        return int(np.floor(len(self.games) / self.batch_size))

    def __choice_data(self, X, y):
        """
        Randomly selects samples from the generated data and applies one-hot encoding to the labels.

        Args:
            X: List of board states.
            y: List of corresponding moves.

        Returns:
            Tuple containing (processed input data, labels in one-hot format).
"""
        sorted_idx = np.random.randint(low=0, high=len(X), size=min(self.moves_per_game, len(X)))

        final_X, final_y = [], []

        for idx in sorted_idx:
            table = np.array(X[idx])
            final_X.append(table)
            sparse_res = np.zeros(4098)
            sparse_res[MOVE_TO_IDX[y[idx]]] = 1
            final_y.append(sparse_res)

        return final_X, final_y

    def __getitem__(self, index):
        """
        Generates a data batch.

        Args:
            index: Index of the batch to be generated.

        Returns:
            Tuple containing (features, labels) for the requested batch.
"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [k for k in indexes]
        game = self.games[list_IDs_temp[0]]
        # Generate data
        X, y = self.__data_generation(game)
        X, y = self.__choice_data(X, y)
        X, y = np.array(X), np.array(y)

        return X, y

    def on_epoch_end(self):
        """
        Updates the indices after each epoch, optionally shuffling the data.
        """

        self.indexes = np.arange(len(self.games))
        np.random.shuffle(self.indexes)

    def __data_generation(self, game):
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

                if self.for_white_generation == is_white:
                    move = move.replace('+', '').replace('#', '').replace('x', '').replace('=', '')

                    if move == 'O-O':
                        pred_move = 'O-O'
                    elif move == 'O-O-O':
                        pred_move = 'O-O-O'
                    else:
                        col, line = find_piece(board_state, move, is_white)
                        pred_move = COLS_CONVERTION[col]+str(line)

                        pred_move = pred_move + move[-2:]
                    if pred_move in VALID_MOVES:
                        X.append(board_state)
                        y.append(pred_move)

            except Exception as e:
                break

        return X, y
