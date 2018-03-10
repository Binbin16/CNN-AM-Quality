import numpy as np
import pickle

from keras.models import model_from_json, load_model
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers import Convolution2D, merge, MaxPooling2D, Input, Add, Concatenate
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

import tensorflow as tf

from generator import DataGenerator

# PARAMETERS
EPOCHS = 1 # Number of iterations on the dataset
LR_VEC = [0.0001] #
BATCH_SIZE = 32
VALD_FRAC = 0.2

IMAGE_SHAPE = (300, 300, 3)
NUM_CLASSES = 3

CLASS_NAMES =['Undermelt', 'JustRight', 'Overmelt']

train_dir = '../../input/train/'
test_dir = '../../input/test/'
result_dir = 'Result/'

NFOLDS = 5 # same as num_batches

#train_files = [data_dir + 'data_batch_' + str(ii) + '.bin' for ii in range(1, num_batches+1)]
#test_file = data_dir + 'test_batch.bin'

def network(input_shape = IMAGE_SHAPE, output_size = NUM_CLASSES, LR = 0.0001):
    '''
    Creates and returns CNN network
    '''
    
    model = Sequential()
    
    model.add(Convolution2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=input_shape)) # TODO remove strides
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    #model.add(MaxPooling2D(pool_size(2, 2), padding='same')) # TODO use this instead
    model.add(Convolution2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
    #model.add(MaxPooling2D(pool_size(2, 2), padding='same')) # TODO use this instead
    # TODO add batch normalization and dropout
    model.add(Flatten())
    model.add(Dense(384, activation='relu'))
    model.add(Dense(192, activation='relu'))
    model.add(Dense(output_size, activation='softmax'))
    
    adam = Adam(lr = LR)
    model.compile(loss='categorical_crossentropy', optimizer=adam)
    
    return model

np.random.seed(0) # for regenerating results

with open(train_dir + 'labels.pkl', 'rb') as f:
    labels = pickle.load(f)

IDs = list(labels.keys())
np.random.shuffle(IDs)

vald_start_index = int(len(IDs) * (1-VALD_FRAC))

partition = {'train':IDs[:vald_start_index], 'validation':IDs[vald_start_index:]}

print(len(partition['train']))
print(len(partition['validation']))

training_generator = DataGenerator(train_dir,
                                  input_shape=IMAGE_SHAPE,
                                  output_size=NUM_CLASSES,
                                  batch_size=BATCH_SIZE).generate(labels, partition['train'])
validation_generator = DataGenerator(train_dir,
                                  input_shape=IMAGE_SHAPE,
                                  output_size=NUM_CLASSES,
                                  batch_size=BATCH_SIZE).generate(labels, partition['validation'])

vald_accuracy = np.zeros_like(LR_VEC)
for i, lr in enumerate(LR_VEC):
    
    print('**************************************')
    print('Running with learning rate ' + str(lr))
    
    model = network(LR = lr)
    model.fit_generator(generator = training_generator,
                        steps_per_epoch = len(partition['train'])//BATCH_SIZE,
                        validation_data = validation_generator,
                        validation_steps = len(partition['validation'])//BATCH_SIZE,
                        epochs = EPOCHS)
    
    model.save(result_dir + 'keras_model_' + str(i) + '.h5')