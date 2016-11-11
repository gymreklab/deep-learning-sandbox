'''
Detect the effect of distance between regulatory region
@An Zheng

Model:
1-layer CNN
fully connected layer
'''


import numpy as np
import h5py
import scipy.io
np.random.seed(1337) # for reproducibility

from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.regularizers import l2, activity_l1
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping

from data_in_silco import load

#######################################
# parameters
#######################################

maxlen = 500
batch_size = 10
embedding_dims = 100
nb_filter = 250
filter_length = 4
hidden_dims = 250
nb_epoch = 5


#######################################
# load data
#######################################
(X_train, y_train), (X_test, y_test) = load.load_onehot()

#######################################
# build model
#######################################
print 'building model'
model = Sequential()
model.add(Convolution1D(input_dim=4,
                        input_length=maxlen,
                        nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode="valid",
                        activation="relu",
                        subsample_length=1))

model.add(MaxPooling1D(pool_length=13, stride=13))

#model.add(Dropout(0.2))
'''
model.add(brnn)

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(input_dim=75*640, output_dim=925))
model.add(Activation('relu'))

model.add(Dense(input_dim=925, output_dim=919))
model.add(Activation('sigmoid'))
'''

model.add(Flatten())
# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
#model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

print 'compiling model'
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


######################################
# train model
######################################
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_test, y_test))


###END#########################################3
