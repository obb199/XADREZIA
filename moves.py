cols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
rows = ['0', '1', '2', '3', '4', '5', '6', '7']

squares = []
for c in cols:
    for r in rows:
        squares.append(c+r)

VALID_MOVES = []
for s1 in squares:
    for s2 in squares:
        VALID_MOVES.append(s1+s2)

VALID_MOVES.append('O-O')
VALID_MOVES.append('O-O-O')
IDX_TO_MOVE = {}  # Dictionary to map index to move string
MOVE_TO_IDX = {}  # Dictionary to map move string to index

idx = 0
for s in VALID_MOVES:
    IDX_TO_MOVE[idx] = s
    MOVE_TO_IDX[s] = idx
    idx += 1
