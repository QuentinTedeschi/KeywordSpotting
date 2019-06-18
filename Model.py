from preprocess import *
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Activation, Input, TimeDistributed, Conv1D
from keras.layers import GRU, BatchNormalization
from keras.optimizers import Adam

Tx = 1101 #Nombre de division en temps
freq = 101 #Nombre de division en fréquence
Ty = 272 #Nombre de division en temps à la sortie

epochs = 50
batch_size = 100
verbose = 1

X = np.load("x_images_arrays.npy")
y = np.load("y_labels.npy")
    
split_ratio = 0.75
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= (1 - split_ratio), random_state=42, shuffle=True)

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

def model(input_shape):
    X_input = Input(shape = input_shape)
    # Première couche : convolution
    X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)                                 # CONV1D
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(0.8)(X)
    # Deuxième couche : GRU
    X = GRU(units=128, return_sequences=True)(X)                         
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)
    # Troisième couche : GRU
    X = GRU(units=128, return_sequences=True)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.8)(X)
    # 4ième couche
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X)
    model = Model(inputs = X_input, outputs = X)    
    return model


model = model(input_shape = (Tx, freq))
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test_hot))
model.save('model.h5')