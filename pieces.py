import numpy as np

base = np.zeros(7)  # Base vector representing empty piece

EMPTY = np.copy(base)  # Empty square representation

# White pieces
WHITE_PAWN = np.copy(base)
WHITE_PAWN[0] = 1
WHITE_PAWN[-1] = 1

WHITE_KNIGHT = np.copy(base)
WHITE_KNIGHT[1] = 1
WHITE_KNIGHT[-1] = 1

WHITE_BISHOP = np.copy(base)
WHITE_BISHOP[2] = 1
WHITE_BISHOP[-1] = 1

WHITE_ROOK = np.copy(base)
WHITE_ROOK[3] = 1
WHITE_ROOK[-1] = 1

WHITE_QUEEN = np.copy(base)
WHITE_QUEEN[4] = 1
WHITE_QUEEN[-1] = 1

WHITE_KING = np.copy(base)
WHITE_KING[5] = 1
WHITE_KING[-1] = 1

# Black pieces
BLACK_PAWN = np.copy(base)
BLACK_PAWN[0] = 1

BLACK_KNIGHT = np.copy(base)
BLACK_KNIGHT[1] = 1

BLACK_BISHOP = np.copy(base)
BLACK_BISHOP[2] = 1

BLACK_ROOK = np.copy(base)
BLACK_ROOK[3] = 1

BLACK_QUEEN = np.copy(base)
BLACK_QUEEN[4] = 1

BLACK_KING = np.copy(base)
BLACK_KING[5] = 1
