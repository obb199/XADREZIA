import model
import numpy as np
import table
import generate_training_data
import move_dict

if __name__ == '__main__':
    Xadrezia = model.Xadrezia()
    Xadrezia(np.random.randn(1, 8, 8, 7))
    Xadrezia.load_weights('weights.weights.h5')

    boardgame = table.generate_start_table()
    generate_training_data.transposition(boardgame, 'e4', True, None)
    generate_training_data.transposition(boardgame, 'e5', False, 'e4')
    generate_training_data.transposition(boardgame, 'Nf3', True, 'e5')
    generate_training_data.transposition(boardgame, 'Nf6', False, 'Nf3')

    out = Xadrezia(np.array([boardgame]))
    out = out[0]
    out = np.squeeze(out)
    out = out.tolist()
    out_1 = out[0:386]
    move = int(np.argmax(out[0:386]))
    print(move)
    #col = np.argmax(out[386:386+9])
    #line = np.argmax(out[386+9:])

    print(move_dict.IDX_TO_MOVE[move])
    #print(col, line)


