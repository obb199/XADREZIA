from generate_training_data import transposition, find_piece, sequence_of_moves, generate_uniques_matches
from table import generate_start_table
from moves import VALID_MOVES
import os
import numpy as np
from convertions import COLS_CONVERTION


def clean_data(for_white_generation):
    FILEPATH = '/home/user/PycharmProjects/chess_engine/matches/'
    matches = [FILEPATH + path for path in os.listdir(FILEPATH)]
    processed_matches = []
    for player_matches in matches:
        processed_matches += generate_uniques_matches(player_matches)

    correct_games = []
    for g in processed_matches:
        moves = sequence_of_moves(g)
        boardgame = generate_start_table()
        # Generate data
        for idx, move in enumerate(moves):
            is_white = True if idx % 2 == 0 else False
            board_state = boardgame.copy()

            try:
                boardgame = transposition(boardgame, move, is_white)

                if for_white_generation == is_white:
                    move = move.replace('+', '').replace('#', '').replace('x', '')

                    if move == 'O-O':
                        correct_games.append(sequence_of_moves(g))
                        break
                    elif move == 'O-O-O':
                        correct_games.append(sequence_of_moves(g))
                        break
                    else:
                        col, line = find_piece(board_state, move, is_white)
                        pred_move = COLS_CONVERTION[col] + str(line)
                        pred_move = pred_move + move[-2:]

                    if pred_move in VALID_MOVES:
                        correct_games.append(sequence_of_moves(g))
                        break

            except Exception as e:
                break

    return correct_games


def shuffle(data, trades, seed=42):
    np.random.seed(seed)
    for _ in range(trades):
        i, j = np.random.randint(low=0, high=len(data)), np.random.randint(low=0, high=len(data))
        data[i], data[j] = data[j], data[i]

    return data
