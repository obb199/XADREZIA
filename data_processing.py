import os
from generate_training_data import transposition
from table import generate_start_table
import numpy as np

FILEPATH = '/home/user/PycharmProjects/chess_engine/matches/'


def generate_uniques_matches(input_path):
    def data_filter(data):
        return True if data[0] != '[' and data[0] != '\n' else False

    with open(input_path, 'r') as f:
        brute_data = f.readlines()
        cleaned_data = list(filter(data_filter, brute_data))
        single_matches = []
        data = ''
        for clean_data in cleaned_data:
            data += clean_data + ' '
            if '1-0' in data or '0-1' in data or '1/2-1/2' in data:
                single_matches.append(data.replace('  1-0', '').replace('  0-1', '').replace('\n', '').replace('1/2-1/2', ''))
                data = ''
    return single_matches


def sequence_of_moves(unique_match):
    unique_match = unique_match.split(' ')
    for idx, move in enumerate(unique_match):
        if '.' in move:
            unique_match[idx] = move.split('.')[1]
    if len(unique_match[-1]) < 2:
        return unique_match[:-1]
    return unique_match


if __name__ == '__main__':
    FILEPATH = '/home/user/PycharmProjects/chess_engine/matches/'
    matches = [FILEPATH + path for path in os.listdir(FILEPATH)]
    processed_matches = []
    for player_matches in matches:
        processed_matches += generate_uniques_matches(player_matches)

    counter_white = 0
    counter_black = 0
    for match_number, m in enumerate(processed_matches):
        match = sequence_of_moves(m)
        boardgame = generate_start_table()
        previous_move = None
        for idx, move in enumerate(match):
            if move == '' or '1-' in move or '-1' in move:
                break
            is_white = idx % 2 == 0
            try:
                boardgame = transposition(boardgame, move, is_white, previous_move)
                if is_white:
                    np.save(f'white_responses/{move}_{counter_white}', boardgame)
                    counter_white += 1
                else:
                    np.save(f'black_responses/{move}_{counter_black}', boardgame)
                    counter_black += 1
                previous_move = move
            except Exception as e:
                print(f"Error processing move {move} in match {match_number}: {e}")
                break
