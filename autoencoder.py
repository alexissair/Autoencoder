import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import keras.backend as K
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import load_model

class Autoencoder() :
    def __init__(self, X, reshape, optimizer, model = None) :
        # The shape of X needs to be (n_samples, n_rows, n_cols) or (n_samples, n_rows, n_cols, 1)
        # If of the shape (n_samples, n_rows, n_cols), reshape needs to be set to True.
        self.X = X
        self.nsamples = self.X.shape[0]
        self.nrows = self.X.shape[1]
        self.ncols = self.X.shape[2]
        self.reshape = reshape
        self.input_shape = (self.nrows, self.ncols, 1)
        if self.reshape :
            self.X = self.X.reshape(self.nsamples, self.nrows, self.ncols, 1)
        if K.image_data_format == 'channels_first' :
            self.X = self.X.reshape(self.nsamples, 1, self.nrows, self.ncols)
            self.input_shape = (1, self.nrows, self.ncols)
        self.X = self.X / 255.
        if model :
            self.model = model
        else :
            self.model = self.get_model()
        self.model.compile(optimizer = optimizer, loss='mse')
        return


    def get_model(self) :
        # The dimentsions are given for samples of shape 28 * 28 * 1.
        model = Sequential()
        model.add(Conv2D(filters = 32, kernel_size=(3, 3), activation='relu', input_shape = self.input_shape, 
                        padding = 'same'))
        # 28*28*32
        model.add(MaxPooling2D(pool_size = (2, 2)))
        # 14*14*32
        model.add(Conv2D(filters = 64, kernel_size=(3, 3), activation='relu',padding = 'same'))
        # 14* 14 * 64
        model.add(MaxPooling2D(pool_size = (2, 2)))
        # 7*7*64
        model.add(Conv2D(filters = 128, kernel_size=(3, 3), activation='relu',padding = 'same'))
        # 7*7*128
        model.add(Conv2D(filters = 64, kernel_size=(3, 3), activation='relu',padding = 'same'))
        # 7 * 7 * 64
        model.add(UpSampling2D(size = (2, 2)))
        # 14 * 14 * 64
        model.add(Conv2D(filters = 32, kernel_size=(3, 3), activation='relu',padding = 'same'))
        # 14 * 14 * 32
        model.add(UpSampling2D(size = (2, 2)))
        # 28 * 28 * 32
        model.add(Conv2D(filters = 1, kernel_size=(3, 3), activation='sigmoid',padding = 'same'))
        return model

    def train(self, epochs, batch_size) :
        self.model.fit(self.X, self.X, epochs=epochs, batch_size=batch_size)
        return
    
    def denoise(self, X) :
        return self.model.predict(X)

    def save(self, path) :
        self.model.save_weights('./'+ path + '.h5')
        return


if __name__ == "__main__":
    # The model was trained on the x_train dataset from MNIST
    (_ , _) , (x_test, _) = mnist.load_data()
    ae = Autoencoder(x_test, reshape = True, optimizer = 'SGD')
    model = ae.get_model()
    model.load_weights('./autoencoder_weights.h5')
    x_test_noisy = ae.X + 0.3*np.random.standard_normal(size = ae.X.shape)
    print(ae.X.shape)
    x_test_denoised = model.predict(x_test_noisy)
    for i in range(10) :
        plt.subplot(211)
        x_test_noisy = x_test_noisy.reshape(x_test_noisy.shape[0], x_test_noisy.shape[1], x_test_noisy.shape[2])
        x_test_denoised = x_test_denoised.reshape(x_test_denoised.shape[0], x_test_denoised.shape[1], x_test_denoised.shape[2])
        plt.imshow(x_test_noisy[i, :, :], cmap='gray')
        plt.subplot(212)
        plt.imshow(x_test_denoised[i, :, :], cmap='gray')
        plt.show()
        # plt.imshow(x_test_denoised[i, :, :, :])
    

