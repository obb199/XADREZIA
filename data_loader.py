import tensorflow as tf
import keras
import numpy as np
from generate_training_data import transposition, find_piece
from table import generate_start_table
from move_dict import MOVE_TO_IDX, VALID_MOVES


class DataGenerator(keras.utils.Sequence):
    """
        Gerador de dados para treinamento de modelos de xadrez.

        Esta classe implementa um gerador de dados que processa partidas de xadrez,
        gera estados de tabuleiro e movimentos correspondentes, e fornece batches
        para treinamento de modelos de aprendizado de máquina.

        Args:
            games: Lista de partidas de xadrez a serem processadas.
            batch_size: Número de partidas por batch (padrão: 1).
            moves_per_game: Número máximo de movimentos a serem considerados por partida (padrão: 16).
            shuffle: Booleano indicando se os dados devem ser embaralhados após cada época (padrão: True).
            for_white_generation: Booleano indicando se deve gerar dados para movimentos das brancas (padrão: True).
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
       Calcula o número de batches por época.

       Returns:
           Número inteiro representando o número de batches por época.
        """
        return int(np.floor(len(self.games) / self.batch_size))

    def __choice_data(self, X, y):
        """
        Seleciona aleatoriamente amostras dos dados gerados e aplica one-hot encoding aos rótulos.

        Args:
            X: Lista de estados de tabuleiro.
            y: Lista de movimentos correspondentes.

        Returns:
            Tupla contendo (dados de entrada processados, rótulos em formato one-hot).
        """
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
        """
        Gera um batch de dados.

        Args:
            index: Índice do batch a ser gerado.

        Returns:
            Tupla contendo (features, labels) para o batch solicitado.
        """
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
        """
        Atualiza os índices após cada época, opcionalmente embaralhando os dados.
        """

        self.indexes = np.arange(len(self.games))
        np.random.shuffle(self.indexes)

    def __data_generation(self, game):
        """
        Gera dados de treinamento a partir de uma partida de xadrez.

        Processa cada movimento da partida, atualiza o estado do tabuleiro e
        armazena os pares (estado, movimento) para treinamento.

        Args:
            game: Partida de xadrez a ser processada.

        Returns:
            Tupla contendo (lista de estados de tabuleiro, lista de movimentos correspondentes).
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
