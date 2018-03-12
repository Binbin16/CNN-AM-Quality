import numpy as np
import pickle

from keras.models import model_from_json, load_model
from keras.models import Sequential, Model
from keras.layers.core import Dense, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Input, Dropout
from keras.regularizers import l1, l2
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras import backend as K

import tensorflow as tf

from generator import DataGenerator

# PARAMETERS
EPOCHS = 64 # Number of iterations on the dataset
LR_VEC = [0.0001] #
BATCH_SIZE = 128
VALD_FRAC = 0.1

IMAGE_SHAPE = (300, 300, 3)
NUM_CLASSES = 3

CLASS_NAMES =['Undermelt', 'JustRight', 'Overmelt']

train_dir = '../../input/train/'
test_dir = '../../input/test/'
result_dir = 'Result/'

#train_files = [data_dir + 'data_batch_' + str(ii) + '.bin' for ii in range(1, num_batches+1)]
#test_file = data_dir + 'test_batch.bin'

def network(input_shape = IMAGE_SHAPE, output_size = NUM_CLASSES, LR = 0.0001):
    '''
    Creates and returns CNN network
    '''
    
    inputLayer = Input(shape = input_shape)
    
    layer = BatchNormalization()(inputLayer)
    layer = Convolution2D(16, (5, 5), padding='same', activation='relu', kernel_regularizer=l2(0.01))(layer)
    layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(layer)
    layer = Convolution2D(32, (5, 5), padding='same', activation='relu', kernel_regularizer=l2(0.01))(layer)
    layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(layer)
    layer = Convolution2D(64, (5, 5), padding='same', activation='relu', kernel_regularizer=l2(0.01))(layer)
    layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(layer)
    
    layer = Flatten()(layer)
    layer = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(layer)
    layer = Dropout(0.25)(layer)
    layer = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(layer)
    layer = Dropout(0.25)(layer)
    
    outputLayer = Dense(output_size, activation='softmax')(layer)
    
    model = Model(inputs=inputLayer, outputs=outputLayer)
    
    adam = Adam(lr = LR)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    return model

seed = 0
np.random.seed(seed) # for regenerating results
tf.set_random_seed(seed)

# TODO: Tensorflow GPU optimization
num_cores = 128
GPU=False
CPU=True
if GPU:
    num_GPU = 2
    num_CPU = 1
if CPU:
    num_CPU = 16
    num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
sess = tf.Session(config=config)
K.set_session(sess)

###--------- TRAINING AND VALIDATION ---------###

# read labels
with open(train_dir + 'labels.pkl', 'rb') as f:
    labels = pickle.load(f)

IDs = list(labels.keys())
np.random.shuffle(IDs)

# partition samples into training and validation
vald_start_index = int(len(IDs) * (1-VALD_FRAC))
partition = {'train':IDs[:vald_start_index], 'validation':IDs[vald_start_index:]}

#print(len(partition['train']))
#print(len(partition['validation']))

# create batch data generators
training_generator = DataGenerator(train_dir,
                                  input_shape=IMAGE_SHAPE,
                                  output_size=NUM_CLASSES,
                                  batch_size=BATCH_SIZE).generate(labels, partition['train'])
validation_generator = DataGenerator(train_dir,
                                  input_shape=IMAGE_SHAPE,
                                  output_size=NUM_CLASSES,
                                  batch_size=BATCH_SIZE).generate(labels, partition['validation'])

# learn model for different learning rates
vald_accuracy = np.zeros_like(LR_VEC)
for i, lr in enumerate(LR_VEC):
    
    print('**************************************')
    print('Running with learning rate ' + str(lr))
    
    model = network(LR = lr)
    history = model.fit_generator(generator = training_generator,
                                  steps_per_epoch = len(partition['train'])//BATCH_SIZE,
                                  validation_data = validation_generator,
                                  validation_steps = len(partition['validation'])//BATCH_SIZE,
                                  epochs = EPOCHS,
                                  verbose = 0)
    
    # save model and results
    model.save(result_dir + 'keras_model_' + str(i) + '.h5')
    with open(result_dir + 'keras_model_' + str(i) + 'history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    
    #print(history.history.keys())
    #print(history.history.values())
    vald_accuracy[i] = history.history['val_acc'][-1] # save final validation accuracy

print('VALIDATION ACCURACY:')
print(vald_accuracy)
print('')

###--------- TESTING ---------###

# find the best LR and load corresponding model : maximum validation accuracy
mx_i = np.argmax(vald_accuracy)
model = load_model(result_dir + 'keras_model_' + str(mx_i) + '.h5')

# read test labels
with open(test_dir + 'labels.pkl', 'rb') as f:
    labels = pickle.load(f)

# create batch data generator from test dataset
IDs = list(labels.keys())
testing_generator = DataGenerator(test_dir,
                                  input_shape=IMAGE_SHAPE,
                                  output_size=NUM_CLASSES,
                                  batch_size=BATCH_SIZE).generate(labels, IDs)

# evaluate the model on test dataset
test_score = model.evaluate_generator(testing_generator,
                                      steps = len(IDs)//BATCH_SIZE)
print('TEST SCORE:')
print(test_score)
print('')