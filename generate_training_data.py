import numpy as np
from pieces import (
    EMPTY, WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_ROOK, WHITE_QUEEN, WHITE_KING,
    BLACK_PAWN, BLACK_KNIGHT, BLACK_BISHOP, BLACK_ROOK, BLACK_QUEEN, BLACK_KING
)
from pieces import NOTE_CONVERTION, WHITE_CONVERTION, BLACK_CONVERTION


def generate_uniques_matches(input_path, only_white=True, only_black=False):
    """
        Processa um arquivo de partidas de xadrez e extrai partidas individuais.

        Args:
            input_path: Caminho para o arquivo contendo as partidas brutas.

        Returns:
            Lista de strings, onde cada string representa uma partida única sem informações de resultado.
        """
    def data_filter(data):
        return True if data[0] != '[' and data[0] != '\n' else False

    with open(input_path, 'r') as f:
        brute_data = f.readlines()
        cleaned_data = list(filter(data_filter, brute_data))
        single_matches = []
        data = ''
        for clean_data in cleaned_data:
            data += clean_data + ' '
            if only_white:
                if '1-0' in data:
                    single_matches.append(data.replace('  1-0', '').replace('  0-1', '').replace('\n', '').replace('1/2-1/2', ''))
                    data = ''
            elif only_black:
                if '0-1' in data:
                    single_matches.append(data.replace('  1-0', '').replace('  0-1', '').replace('\n', '').replace('1/2-1/2', ''))
                    data = ''
            else:
                if '1-0' in data or '0-1' in data or '1/2-1/2' in data:
                    single_matches.append(
                        data.replace('  1-0', '').replace('  0-1', '').replace('\n', '').replace('1/2-1/2', ''))
                    data = ''

    return single_matches


def sequence_of_moves(unique_match):
    """
        Extrai a sequência de movimentos de uma partida de xadrez.

        Args:
            unique_match: String contendo uma partida completa com movimentos.

        Returns:
            Lista de movimentos individuais limpos, sem números de turno.
        """
    unique_match = unique_match.split(' ')
    for idx, move in enumerate(unique_match):
        if '.' in move:
            unique_match[idx] = move.split('.')[1]
    if len(unique_match[-1]) < 2:
        return unique_match[:-1]
    return unique_match


def compare_pieces(piece_1: np.ndarray, piece_2: np.ndarray) -> bool:
    """
    Compara duas peças para verificar se são iguais.

    Args:
        piece_1: Vetor representando a primeira peça.
        piece_2: Vetor representando a segunda peça.

    Returns:
        True se as peças forem iguais, False caso contrário.
    """
    return np.array_equal(piece_1, piece_2)


def find_kings(table):
    """
        Localiza as posições dos reis branco e preto no tabuleiro.

        Args:
            table: Tabuleiro de xadrez representado como um array numpy 8x8x7.

        Returns:
            Tupla contendo ((col_white, row_white), (col_black, row_black)) com as posições dos reis.
        """
    white_king_position, black_king_position = None, None

    for i in range(8):
        for j in range(8):
            if white_king_position is None or black_king_position is None:
                if compare_pieces(table[i, j], WHITE_KING):
                    white_king_position = i, j
                elif compare_pieces(table[i, j], BLACK_KING):
                    black_king_position = i, j
            else:
                break

    return white_king_position, black_king_position


def find_controlled_squares(table, i, j):
    """
        Identifica todos os quadrados controlados por uma peça específica.

        Args:
            table: Tabuleiro de xadrez 8x8x7.
            i: Índice da coluna da peça.
            j: Índice da linha da peça.

        Returns:
            Tupla contendo (lista de quadrados controlados, informações da peça).
        """
    if compare_pieces(table[i, j], WHITE_PAWN) or compare_pieces(table[i, j], BLACK_PAWN):
        return pawn_control(i, j, True) if table[i, j, -1] == 1 else pawn_control(i, j, False)
    if compare_pieces(table[i, j, :-1], WHITE_KNIGHT[:-1]):
        return knight_control(i, j)
    if compare_pieces(table[i, j, :-1], WHITE_BISHOP[:-1]):
        return bishop_control(table, i, j)
    if compare_pieces(table[i, j, :-1], WHITE_ROOK[:-1]):
        return rook_control(table, i, j)
    if compare_pieces(table[i, j, :-1], WHITE_QUEEN[:-1]):
        return queen_control(table, i, j)
    if compare_pieces(table[i,j, :-1], WHITE_KING[:-1]):
        return king_control(i, j)


def pawn_control(col, row, is_white):
    """
        Calcula os quadrados controlados por um peão.

        Args:
            col: Coluna atual do peão.
            row: Linha atual do peão.
            is_white: Booleano indicando se o peão é branco.

        Returns:
            Tupla (quadrados controlados, ['pawn', col, row]).
        """
    controlled_squares_list = []
    if is_white:
        if col + 1 <= 7 and row - 1 >= 0:
            controlled_squares_list.append([col + 1, row - 1])
        if col + 1 <= 7 <= row + 1:
            controlled_squares_list.append([col + 1, row + 1])
    else:
        if col - 1 >= 7 and row - 1 >= 0:
            controlled_squares_list.append([col + 1, row - 1])
        if col - 1 >= 7 and 7 <= row + 1:
            controlled_squares_list.append([col + 1, row + 1])

    return controlled_squares_list, ['pawn', col, row]


def bishop_control(table, col, row):
    """
        Calcula os quadrados controlados por um bispo.

        Args:
            table: Tabuleiro de xadrez.
            col: Coluna atual do bispo.
            row: Linha atual do bispo.

        Returns:
            Tupla (quadrados controlados, ['bishop', col, row]).
        """

    controlled_squares_list = []

    directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    for dr, dc in directions:
        r, c = row + dr, col + dc
        while 0 <= r <= 7 and 0 <= c <= 7:
            controlled_squares_list.append([c, r])
            if not compare_pieces(table[c, r], EMPTY):
                break
            r += dr
            c += dc

    return controlled_squares_list, ['bishop', col, row]


def knight_control(i, j):
    """
        Calcula os quadrados controlados por um cavalo.

        Args:
            i: Coluna atual do cavalo.
            j: Linha atual do cavalo.

        Returns:
            Tupla (quadrados controlados, ['knight', i, j]).
        """
    controlled_squares_list = []
    pos = [(i + 2, j + 1), (i + 2, j - 1),
           (i - 2, j + 1), (i - 2, j - 1),
           (i + 1, j + 2), (i + 1, j - 2),
           (i - 1, j + 2), (i - 1, j - 2)]

    for c, l in pos:
        if 0 <= c <= 7 and 0 <= l <= 7:
            controlled_squares_list.append([c, l])

    return controlled_squares_list, ['knight', i, j]


def rook_control(table, col, row):
    """
        Calcula os quadrados controlados por uma torre.

        Args:
            table: Tabuleiro de xadrez.
            col: Coluna atual da torre.
            row: Linha atual da torre.

        Returns:
            Tupla (quadrados controlados, ['rook', col, row]).
        """
    controlled_squares_list = []
    for x in range(row + 1, 8):
        controlled_squares_list.append([col, x])
        if not compare_pieces(table[col, x], EMPTY):
            break

    for x in range(row - 1, -1, -1):
        controlled_squares_list.append([col, x])
        if not compare_pieces(table[col, x], EMPTY):
            break

    for x in range(col + 1, 8):
        controlled_squares_list.append([x, row])
        if not compare_pieces(table[x, row], EMPTY):
            break

    for x in range(col - 1, -1, -1):
        controlled_squares_list.append([x, row])
        if not compare_pieces(table[x, row], EMPTY):
            break

    return controlled_squares_list, ['rook', col, row]


def king_control(col, row):
    """
        Calcula os quadrados controlados por um rei.

        Args:
            col: Coluna atual do rei.
            row: Linha atual do rei.

        Returns:
            Tupla (quadrados controlados, ['king', col, row]).
        """
    controlled_squares_list = []
    for x in [0, 1, -1]:
        for y in [0, 1, -1]:
            if x == y and y == 0:
                continue
            if 0 <= row + y <= 7 and 0 <= col + x <= 7:
                controlled_squares_list.append([col + x, row + y])

    return controlled_squares_list, ['king', col, row]


def queen_control(table, col, row):
    """
        Calcula os quadrados controlados por uma rainha.

        Args:
            table: Tabuleiro de xadrez.
            col: Coluna atual da rainha.
            row: Linha atual da rainha.

        Returns:
            Tupla (quadrados controlados, ['queen', col, row]).
        """

    bishop_like_control = list(bishop_control(table, col, row)[0])
    rook_like_control = list(rook_control(table, col, row)[0])
    controlled_squares_list = bishop_like_control + rook_like_control

    return controlled_squares_list, ['queen', col, row]


def verify_check(table):
    """
        Verifica se algum rei está em xeque.

        Args:
            table: Tabuleiro de xadrez 8x8x7.

        Returns:
            Tupla (white_in_check, black_in_check) indicando quais reis estão em xeque.
        """
    whites_in_check, blacks_in_check = False, False

    white_king_position, black_king_position = find_kings(table)

    for row in range(8):
        for col in range(8):
            if not compare_pieces(table[col, row], EMPTY):
                output = find_controlled_squares(table, col, row)
                controlled_squares = output[0]
                if (table[col, row, -1] == 1 and blacks_in_check) or (table[col, row, -1] == 0 and whites_in_check):
                    continue

                if table[col, row, -1] == 1:
                    if list(black_king_position) in controlled_squares:
                        blacks_in_check = True
                elif table[col, row, -1] == 0:
                    if list(white_king_position) in controlled_squares:
                        whites_in_check = True

    return whites_in_check, blacks_in_check


def find_piece(table: np.ndarray, move: str, is_white: bool) -> tuple[int, int]:
    """
    Encontra a posição de uma peça no tabuleiro com base no movimento.

    Args:
        table: Tabuleiro de xadrez (8x8x7).
        move: Movimento em notação algébrica (ex: "e4", "Nf3").
        is_white: True se a peça é branca, False se preta.

    Returns:
        Tuple com coordenadas (col, line) da peça, or None if not found.
    """
    # Pawn search
    if move[0].islower():
        piece = WHITE_PAWN if is_white else BLACK_PAWN
        if 'x' in move:  # Diagonal capture (e.g., "dxc6")
            init_col = NOTE_CONVERTION[move[0]]
            dest_line = int(move[-1]) - 1
            start_line = dest_line - 1 if is_white else dest_line + 1
            if 0 <= start_line <= 7 and compare_pieces(table[init_col, start_line], piece):
                return init_col, start_line
        else:  # Non-capture pawn move (e.g., "e4")
            col = NOTE_CONVERTION[move[0]]
            line = int(move[-1]) - 1
            start_line = line - 1 if is_white else line + 1
            if 0 <= start_line <= 7 and compare_pieces(table[col, start_line], piece):
                return col, start_line
            start_line = line - 2 if is_white else line + 2
            if 0 <= start_line <= 7 and compare_pieces(table[col, start_line], piece):
                return col, start_line

    # Non-pawn moves
    move_clean = move.replace('x', '').replace('=', '').replace('+', '')
    line = int(move_clean[-1]) - 1
    col = NOTE_CONVERTION[move_clean[-2]]

    col_to_search = None
    line_to_search = None
    if len(move_clean) >= 4 and move_clean[1] in 'abcdefgh':
        col_to_search = NOTE_CONVERTION[move_clean[1]]
    elif len(move_clean) >= 4 and move_clean[1] in '12345678':
        line_to_search = int(move_clean[1]) - 1

    # King's search
    if 'K' in move:
        piece = WHITE_KING if is_white else BLACK_KING
        sums = [[0, 1], [0, -1], [1, 0], [1, -1], [1, 1], [-1, 0], [-1, 1], [-1, -1]]
        for s in sums:
            c, l = col + s[0], line + s[1]
            if 0 <= c <= 7 and 0 <= l <= 7:
                if compare_pieces(table[c, l], piece):
                    return c, l

    # Knight's search
    if 'N' in move:
        piece = WHITE_KNIGHT if is_white else BLACK_KNIGHT
        pos = [(col + 2, line + 1), (col + 2, line - 1),
               (col - 2, line + 1), (col - 2, line - 1),
               (col + 1, line + 2), (col + 1, line - 2),
               (col - 1, line + 2), (col - 1, line - 2)]

        possible_results = []
        for c, l in pos:
            if 0 <= c <= 7 and 0 <= l <= 7:
                if compare_pieces(table[c, l], piece):
                    if col_to_search is None and line_to_search is None:
                        possible_results.append([c, l])
                    if col_to_search is not None and c == col_to_search:
                        possible_results.append([c, l])
                    if line_to_search is not None and l == line_to_search:
                        possible_results.append([c, l])

        if len(possible_results) == 1:
            return possible_results[0][0], possible_results[0][1]
        elif len(possible_results) >= 2:
            for possible_result in possible_results:
                temp_table = table.copy()
                temp_table[col, line] = WHITE_KNIGHT if is_white else BLACK_KNIGHT
                temp_table[possible_result[0], possible_result[1]] = EMPTY
                is_check = verify_check(temp_table)[0] if is_white else verify_check(temp_table)[1]
                if not is_check:
                    return possible_result[0], possible_result[1]

    # Rook's search
    if 'R' in move:
        piece = WHITE_ROOK if is_white else BLACK_ROOK
        if col_to_search is not None:
            if 0 <= line <= 7 and compare_pieces(table[col_to_search, line], piece):
                return col_to_search, line
        if line_to_search is not None:
            if 0 <= col <= 7 and compare_pieces(table[col, line_to_search], piece):
                return col, line_to_search

        possible_results = []
        # Search column
        for i in range(line + 1, 8):
            if compare_pieces(table[col, i], piece):
                possible_results.append([col, i])
            if not compare_pieces(table[col, i], EMPTY):
                break
        for i in range(line - 1, -1, -1):
            if compare_pieces(table[col, i], piece):
                possible_results.append([col, i])
            if not compare_pieces(table[col, i], EMPTY):
                break

        # Search row
        for i in range(col + 1, 8):
            if compare_pieces(table[i, line], piece):
                possible_results.append([i, line])
            if not compare_pieces(table[i, line], EMPTY):
                break
        for i in range(col - 1, -1, -1):
            if compare_pieces(table[i, line], piece):
                possible_results.append([i, line])
            if not compare_pieces(table[i, line], EMPTY):
                break

        if len(possible_results) == 1:
            return possible_results[0][0], possible_results[0][1]
        elif len(possible_results) >= 2:
            for possible_result in possible_results:
                temp_table = table.copy()
                temp_table[col, line] = WHITE_ROOK if is_white else BLACK_ROOK
                temp_table[possible_result[0], possible_result[1]] = EMPTY
                is_check = verify_check(temp_table)[0] if is_white else verify_check(temp_table)[1]
                if not is_check:
                    return possible_result[0], possible_result[1]

    # Bishop's search
    if 'B' in move:
        piece = WHITE_BISHOP if is_white else BLACK_BISHOP
        # Check all four diagonal directions
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dc, dl in directions:
            c, l = col, line
            steps = 0
            while 0 <= c + dc <= 7 and 0 <= l + dl <= 7:
                c += dc
                l += dl
                steps += 1
                if compare_pieces(table[c, l], piece):
                    for i in range(1, steps):
                        path_c = col + i * dc
                        path_l = line + i * dl
                        if not compare_pieces(table[path_c, path_l], EMPTY):
                            break
                    if (col_to_search is None or c == col_to_search) and (
                            line_to_search is None or l == line_to_search):
                        return c, l

    # Queen's search
    if 'Q' in move:
        piece = WHITE_QUEEN if is_white else BLACK_QUEEN
        c = 0
        positions = []
        for i in range(8):
            for j in range(8):
                if compare_pieces(table[i, j], piece):
                    c += 1
                    positions.append([i, j])
        if c == 1:
            return positions[0][0], positions[0][1]

        if c > 1:
            if col_to_search is not None:
                for i in range(8):
                    if compare_pieces(table[col_to_search, i], piece):
                        return col_to_search, i
            if line_to_search is not None:
                for i in range(8):
                    if compare_pieces(table[i, line_to_search], piece):
                        return i, line_to_search

            control = queen_control(table, col, line)[0]
            for p in positions:
                if p in control:
                    return p


def transposition(table: np.ndarray, move: str, is_white: bool, previous_move: str = None) -> np.ndarray:
    """
    Atualiza o tabuleiro com base no movimento.

    Args:
        table: Tabuleiro de xadrez (8x8x7).
        move: Movimento em notação algébrica.
        is_white: True se a peça é branca, False se preta.
        previous_move: Lance anterior.

    Returns:
        Tabuleiro atualizado.
    """
    if 'O-O-O' in move:
        if is_white:
            if not (compare_pieces(table[4, 0], WHITE_KING) and compare_pieces(table[0, 0], WHITE_ROOK)):
                raise ValueError("Invalid castling: King or rook not in position.")
            table[2, 0] = WHITE_KING
            table[3, 0] = WHITE_ROOK
            table[4, 0] = EMPTY
            table[0, 0] = EMPTY
        else:
            if not (compare_pieces(table[4, 7], BLACK_KING) and compare_pieces(table[0, 7], BLACK_ROOK)):
                raise ValueError("Invalid castling: King or rook not in position.")
            table[2, 7] = BLACK_KING
            table[3, 7] = BLACK_ROOK
            table[4, 7] = EMPTY
            table[0, 7] = EMPTY
        return table
    # Handle castling

    if 'O-O' in move:
        if is_white:
            if not (compare_pieces(table[4, 0], WHITE_KING) and compare_pieces(table[7, 0], WHITE_ROOK)):
                raise ValueError("Invalid castling: King or rook not in position.")
            table[6, 0] = WHITE_KING
            table[5, 0] = WHITE_ROOK
            table[4, 0] = EMPTY
            table[7, 0] = EMPTY
        else:
            if not (compare_pieces(table[4, 7], BLACK_KING) and compare_pieces(table[7, 7], BLACK_ROOK)):
                raise ValueError("Invalid castling: King or rook not in position.")
            table[6, 7] = BLACK_KING
            table[5, 7] = BLACK_ROOK
            table[4, 7] = EMPTY
            table[7, 7] = EMPTY
        return table

    move = move.replace('+', '')  # Ignore check notation

    # Handle pawn promotion
    if '=' in move:
        if len(move) == 4:  # Promotion without capture (e.g., "e8=Q")
            line = int(move[1]) - 1
            col = NOTE_CONVERTION[move[0]]
            promoted_piece = WHITE_CONVERTION[move[-1]] if is_white else BLACK_CONVERTION[move[-1]]

            if (is_white and line != 7) or (not is_white and line != 0):
                raise ValueError("Invalid promotion: Pawn not on promotion rank.")

            start_line = line - 1 if is_white else line + 1
            if not (0 <= start_line <= 7 and compare_pieces(table[col, start_line],
                                                            WHITE_PAWN if is_white else BLACK_PAWN)):
                raise ValueError("Invalid promotion: No pawn to promote.")

            table[col, start_line] = EMPTY
            table[col, line] = promoted_piece
            return table
        else:  # Promotion with capture (e.g., "dxe8=Q")
            start_col = NOTE_CONVERTION[move[0]]
            col = NOTE_CONVERTION[move[2]]
            line = int(move[3]) - 1
            promoted_piece = WHITE_CONVERTION[move[-1]] if is_white else BLACK_CONVERTION[move[-1]]
            piece = WHITE_PAWN if is_white else BLACK_PAWN

            start_line = line - 1 if is_white else line + 1
            if not (0 <= start_line <= 7 and compare_pieces(table[start_col, start_line], piece)):
                raise ValueError("Invalid promotion: No pawn to promote.")

            table[start_col, start_line] = EMPTY
            table[col, line] = promoted_piece
            return table

    # Handle pawn moves
    if move[0].islower():
        piece = WHITE_PAWN if is_white else BLACK_PAWN
        if len(move) == 2:  # Non-capture pawn move (e.g., "e4")
            col = NOTE_CONVERTION[move[0]]
            line = int(move[1]) - 1

            if not compare_pieces(table[col, line], EMPTY):
                raise ValueError(f"Invalid pawn move: Destination {move} is occupied.")

            # Find the pawn (single or double move)
            start_line = line - 1 if is_white else line + 1
            if 0 <= start_line <= 7 and compare_pieces(table[col, start_line], piece):
                table[col, start_line] = EMPTY
                table[col, line] = piece
            else:
                start_line = line - 2 if is_white else line + 2
                if 0 <= start_line <= 7 and compare_pieces(table[col, start_line], piece):
                    table[col, start_line] = EMPTY
                    table[col, line] = piece
                else:
                    raise ValueError(f"Invalid pawn move: No pawn can move to {move}.")
            return table

        else:  # Capture move (e.g., "dxe6")
            start_col = NOTE_CONVERTION[move[0]]
            col = NOTE_CONVERTION[move[-2]]
            line = int(move[-1]) - 1

            # Handle en passant
            if compare_pieces(table[col, line], EMPTY):
                previous_move = previous_move.replace('+', '')
                # En passant: destination is empty, captured pawn is on the same rank
                if is_white and line == 5 and previous_move and len(previous_move) == 2:
                    prev_col = NOTE_CONVERTION[previous_move[0]]
                    prev_line = int(previous_move[1]) - 1
                    if prev_line == 4 and prev_col == col and abs(start_col - col) == 1:
                        table[start_col, line - 1] = EMPTY  # Remove moving pawn
                        table[col, line - 1] = EMPTY  # Remove captured pawn
                        table[col, line] = WHITE_PAWN
                        return table
                elif not is_white and line == 2 and previous_move and len(previous_move) == 2:
                    prev_col = NOTE_CONVERTION[previous_move[0]]
                    prev_line = int(previous_move[1]) - 1
                    if prev_line == 3 and prev_col == col and abs(start_col - col) == 1:
                        table[start_col, line + 1] = EMPTY  # Remove moving pawn
                        table[col, line + 1] = EMPTY  # Remove captured pawn
                        table[col, line] = BLACK_PAWN
                        return table
                raise ValueError(f"Invalid pawn capture: No piece to capture at {move[-2]}{move[-1]}.")

            # Regular pawn capture
            start_line = line - 1 if is_white else line + 1
            if not (0 <= start_line <= 7 and compare_pieces(table[start_col, start_line], piece)):
                raise ValueError(f"Invalid pawn capture: No pawn at {move[0]}{start_line + 1}.")

            if table[col, line][6] == (1 if is_white else 0):
                raise ValueError(f"Invalid capture: Cannot capture own piece at {move[-2]}{move[-1]}.")

            table[start_col, start_line] = EMPTY
            table[col, line] = piece
            return table

    # Non-pawn move
    move_clean = move.replace('x', '')
    line = int(move_clean[-1]) - 1
    col = NOTE_CONVERTION[move_clean[-2]]
    piece_position = 0

    piece = WHITE_CONVERTION[move[piece_position]] if is_white else BLACK_CONVERTION[move[piece_position]]
    previous_pos = find_piece(table, move, is_white)
    if previous_pos is None:
        raise ValueError(
            f"Invalid move: {move_clean}. No {('white' if is_white else 'black')} piece can move to {move_clean[-2]}{move_clean[-1]}.")

    # Validate capture
    if 'x' in move:
        if compare_pieces(table[col, line], EMPTY):
            raise ValueError(f"Invalid capture: No piece to capture at {move_clean[-2]}{move_clean[-1]}.")
        if table[col, line][6] == (1 if is_white else 0):
            raise ValueError(f"Invalid capture: Cannot capture own piece at {move_clean[-2]}{move_clean[-1]}.")
    else:
        if not compare_pieces(table[col, line], EMPTY):
            raise ValueError(f"Invalid move: Destination {move_clean[-2]}{move_clean[-1]} is occupied.")

    table[previous_pos[0], previous_pos[1]] = EMPTY
    table[col, line] = piece

    return table
