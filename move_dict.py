cols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
rows = ['1', '2', '3', '4', '5', '6', '7', '8']
pieces = ['', 'N', 'B', 'R', 'Q', 'K']

IDX_TO_MOVE = {}
MOVE_TO_IDX = {}
VALID_MOVES = []

idx = 0
for c in cols:
    for r in rows:
        for p in pieces:
            IDX_TO_MOVE[idx] = p + c + r
            MOVE_TO_IDX[p+c+r] = idx
            VALID_MOVES.append(p+c+r)
            idx += 1

MOVE_TO_IDX['O-O'] = len(MOVE_TO_IDX)
IDX_TO_MOVE[len(MOVE_TO_IDX)] = 'O-O'

MOVE_TO_IDX['O-O-O'] = len(MOVE_TO_IDX)
IDX_TO_MOVE[len(MOVE_TO_IDX)] = 'O-O-O'

VALID_MOVES.append('O-O')
VALID_MOVES.append('O-O-O')
