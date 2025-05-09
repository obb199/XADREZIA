from pieces import *

# Dictionaries to convert notation to vector representation
WHITE_CONVERTION = {'N': WHITE_KNIGHT,
                    'B': WHITE_BISHOP,
                    'R': WHITE_ROOK,
                    'Q': WHITE_QUEEN,
                    'K': WHITE_KING}

BLACK_CONVERTION = {'N': BLACK_KNIGHT,
                    'B': BLACK_BISHOP,
                    'R': BLACK_ROOK,
                    'Q': BLACK_QUEEN,
                    'K': BLACK_KING}

NOTE_CONVERTION = {'a': 0,
                   'b': 1,
                   'c': 2,
                   'd': 3,
                   'e': 4,
                   'f': 5,
                   'g': 6,
                   'h': 7}

COLS_CONVERTION = {0: 'a',
                   1: 'b',
                   2: 'c',
                   3: 'd',
                   4: 'e',
                   5: 'f',
                   6: 'g',
                   7: 'h'}

PIECE_TO_LETTER = {'king': 'K',
                   'rook': 'R',
                   'bishop': 'B',
                   'queen': 'Q',
                   'knight': 'N'}
