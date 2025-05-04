import model
import utils
import keras
import data_loader
import numpy as np

if __name__ == '__main__':
    data = utils.clean_data(True)
    data = utils.shuffle(data, 10000000)
    gen_train = data_loader.DataGenerator(data[:int(len(data)*0.7)])
    gen_val = data_loader.DataGenerator(data[int(len(data)*0.7):int(len(data)*0.9)])
    gen_test = data_loader.DataGenerator(data[int(len(data)*0.9):])
    Xadrezia = model.Xadrezia()
    Xadrezia.predict(np.random.randn(1, 8, 8, 7), verbose=0)
    Xadrezia.summary()
    #Xadrezia.load_weights('weights.weights.h5')
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    Xadrezia.compile(optimizer=optimizer, loss='categorical_crossentropy')
    Xadrezia.fit(gen_train, epochs=4, validation_data=gen_val)


