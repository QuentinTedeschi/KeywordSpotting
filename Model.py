from preprocess import *
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Activation, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
import os
save_data_to_array(max_len = feature_dim_2)

feature_dim_1 = 20
channel = 1
epochs = 50
batch_size = 100
verbose = 1
num_classes = 10
Tx = 1102
feature_dim_2 = 11
n_freq = 101
Ty = 272


X_train, X_test, y_train, y_test = get_train_test()
X_train = X_train.reshape(X_train.shape[0], feature_dim_1, feature_dim_2, channel)
X_test = X_test.reshape(X_test.shape[0], feature_dim_1, feature_dim_2, channel)

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

def model(input_shape):
    X_input = Input(shape = input_shape)
    # Première couche : convolution
    X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)                                 # CONV1D
#    X = BatchNormalization()(X)
#    X = Activation('relu')(X)
#    X = Dropout(0.8)(X)
    # Deuxième couche : GRU
    X = GRU(units=128, return_sequences=True)(X)                         
    #    X = Dropout(0.8)(X)
#    X = BatchNormalization()(X)
    # Troisième couche : GRU
    X = GRU(units=128, return_sequences=True)(X)
#    X = Dropout(0.8)(X)
#    X = BatchNormalization()(X)
#    X = Dropout(0.8)(X)
    # 4ième couche
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X)
    model = Model(inputs = X_input, outputs = X)    
    return model

#def predict(filepath, model):
#    sample = spectrogram(filepath)
#    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
#    return get_labels()[0][np.argmax(model.predict(sample_reshaped))]
#
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
model = model(input_shape = (Tx, n_freq))
#opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
#model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
#model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test_hot))
#model.save('model.h5')
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)