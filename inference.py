from generate_training_data import find_controlled_squares, compare_pieces, transposition, verify_check
from pieces import EMPTY, WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_ROOK, WHITE_QUEEN, WHITE_KING, BLACK_PAWN, BLACK_ROOK
from table import generate_start_table
from moves import IDX_TO_MOVE
from convertions import COLS_CONVERTION, PIECE_TO_LETTER
from model import Xadrezia
import numpy as np


def get_valid_moves(boardgame, is_white):
    friendly_bit = 1 if is_white else 0
    enemy_bit = 0 if is_white else 1

    moves = []

    for i in range(8):
        for j in range(8):
            if boardgame[i, j, -1] == friendly_bit and not compare_pieces(boardgame[i,j], EMPTY):
                squares, piece_info = find_controlled_squares(boardgame, i, j)
                piece, col, row = piece_info
                init_pos = COLS_CONVERTION[i] + str(j)

                if piece == 'king':  # Castling
                    if i == 4 and (j == 0 or j == 7):
                        rook = WHITE_ROOK if is_white else BLACK_ROOK
                        col = j
                        if compare_pieces(boardgame[col, 0], rook):
                            if compare_pieces(boardgame[col, 0], EMPTY):
                                if compare_pieces(boardgame[col, 0], EMPTY):
                                    if compare_pieces(boardgame[col, 0], EMPTY):
                                        moves.append(['O-O-O'])
                        if compare_pieces(boardgame[col, 6], rook):
                            if compare_pieces(boardgame[col, 5], EMPTY):
                                moves.append(['O-O'])

                elif piece == 'pawn':
                    test_board = boardgame.copy()
                    if friendly_bit == 1:
                        if compare_pieces(boardgame[i, 3], EMPTY):
                            test_board[i, j] = EMPTY
                            test_board[i, j+2] = WHITE_PAWN
                            if verify_check(test_board)[0] is False:
                                moves.append(init_pos + COLS_CONVERTION[i] + str(j+2))

                        if compare_pieces(boardgame[i, j+1], EMPTY):
                            if j+1 <= 7:
                                test_board[i, j] = EMPTY
                                test_board[i, j + 1] = WHITE_PAWN
                                if not verify_check(test_board)[0]:
                                    moves.append(init_pos+COLS_CONVERTION[i]+str(j+1))
                    else:
                        if compare_pieces(boardgame[i, 5], EMPTY):
                            test_board[i, j] = EMPTY
                            test_board[i, j-2] = BLACK_PAWN
                            if verify_check(test_board)[1] is False:
                                moves.append(init_pos + COLS_CONVERTION[i] + str(j-2))
                        if compare_pieces(boardgame[i, j-1], EMPTY):
                            if j-1 >= 0:
                                test_board[i, j - 1] = BLACK_PAWN
                                if verify_check(test_board)[1] is False:
                                    moves.append(init_pos+COLS_CONVERTION[i]+str(j-1))

                for square in squares:  # Any other move beyond castling or paw's movement
                    if (boardgame[square[0], square[1], -1] == enemy_bit and not compare_pieces(boardgame[square[0], square[1]], EMPTY)) or compare_pieces(boardgame[square[0], square[1]], EMPTY):
                        move = PIECE_TO_LETTER[piece]+COLS_CONVERTION[square[0]]+str(square[1]+1)
                        test_board = boardgame.copy()
                        try:
                            transposition(test_board, move, is_white, None)  # raise an error if move is invalid
                            moves.append(init_pos + COLS_CONVERTION[square[0]] + str(square[1]))
                        except ValueError:
                            continue

    return moves


def ranking_moves(move_probabilities, n_moves, board, is_white):
    valid_moves = get_valid_moves(board, is_white)
    move_ranking = []
    parsed_moves = []
    probabilities = []
    for _ in range(n_moves):
        idx_best_move = np.argmax(move_probabilities)
        best_move = IDX_TO_MOVE[idx_best_move]
        if best_move in valid_moves:
            move = IDX_TO_MOVE[idx_best_move]
            move_ranking.append(move)
            probabilities.append(str(move_probabilities[idx_best_move]*100)+'%')
            parsed_moves.append(move_parser(move, board))
        move_probabilities[idx_best_move] = -1

    return move_ranking, probabilities


def move_parser(move, board):
    col_init, row_init = move[0], move[1]
    col_dest, row_dest = move[2], str(int(move[3])+1)
    if compare_pieces(board[col_init, row_init, :-1], WHITE_PAWN[:-1]):
        return col_dest + row_dest
    if compare_pieces(board[col_init, row_init, :-1], WHITE_KNIGHT[:-1]):
        return 'N' + col_dest + row_dest
    if compare_pieces(board[col_init, row_init, :-1], WHITE_BISHOP[:-1]):
        return 'B' + col_dest + row_dest
    if compare_pieces(board[col_init, row_init, :-1], WHITE_ROOK[:-1]):
        return 'R' + col_dest + row_dest
    if compare_pieces(board[col_init, row_init, :-1], WHITE_QUEEN[:-1]):
        return 'Q' + col_dest + row_dest
    if compare_pieces(board[col_init, row_init, :-1], WHITE_KING[:-1]):
        return 'K' + col_dest + row_dest


if __name__ == '__main__':
    t = generate_start_table()
    print(get_valid_moves(t, False))
