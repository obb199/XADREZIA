import model
import utils
import data_loader
import keras


if __name__ == '__main__':
    training_data = utils.clean_data(True)
    training_data = utils.shuffle(training_data, 1000000)
    length_data = len(training_data)

    Xadrezia = model.Xadrezia()
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    Xadrezia.compile(optimizer=optimizer, loss='categorical_crossentropy')
    gen_train = data_loader.DataGenerator(training_data[0:int(length_data * 0.7)], batch_size=1, moves_per_game=16)
    gen_val = data_loader.DataGenerator(training_data[int(length_data * 0.7):int(length_data * 0.9)])
    gen_test = data_loader.DataGenerator(training_data[int(length_data * 0.9):])
    Xadrezia.fit(gen_train, epochs=8, validation_data=gen_val)
    Xadrezia.save_weights('weights.weights.h5')
