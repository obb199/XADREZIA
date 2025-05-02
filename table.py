from pieces import *


def generate_start_table():
    table = np.zeros([8, 8, 7], dtype='float16')
    table[0, 0] = WHITE_ROOK
    table[7, 0] = WHITE_ROOK
    table[1, 0] = WHITE_KNIGHT
    table[6, 0] = WHITE_KNIGHT
    table[2, 0] = WHITE_BISHOP
    table[5, 0] = WHITE_BISHOP
    table[3, 0] = WHITE_QUEEN
    table[4, 0] = WHITE_KING
    table[:, 1] = WHITE_PAWN

    table[0, -1] = BLACK_ROOK
    table[7, -1] = BLACK_ROOK
    table[1, -1] = BLACK_KNIGHT
    table[6, -1] = BLACK_KNIGHT
    table[2, -1] = BLACK_BISHOP
    table[5, -1] = BLACK_BISHOP
    table[3, -1] = BLACK_QUEEN
    table[4, -1] = BLACK_KING
    table[:, 6] = BLACK_PAWN

    return table


def show_pieces(table):
    converted_table = np.full([8, 8], 'xx')
    for i in range(8):
        for j in range(8):
            if np.sum(table[i, j]) == 0:
                continue

            is_white = table[i, j][6] == 1
            piece_type = np.argmax(table[i, j, :-1])

            if is_white:
                if piece_type == 0:
                    converted_table[i, j] = 'wp'
                elif piece_type == 1:
                    converted_table[i, j] = 'wn'
                elif piece_type == 2:
                    converted_table[i, j] = 'wb'
                elif piece_type == 3:
                    converted_table[i, j] = 'wr'
                elif piece_type == 4:
                    converted_table[i, j] = 'wq'
                elif piece_type == 5:
                    converted_table[i, j] = 'wk'
            else:
                if piece_type == 0:
                    converted_table[i, j] = 'bp'
                elif piece_type == 1:
                    converted_table[i, j] = 'bn'
                elif piece_type == 2:
                    converted_table[i, j] = 'bb'
                elif piece_type == 3:
                    converted_table[i, j] = 'br'
                elif piece_type == 4:
                    converted_table[i, j] = 'bq'
                elif piece_type == 5:
                    converted_table[i, j] = 'bk'

    return converted_table
