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
EPOCHS = 5 # Number of iterations on the dataset
LR_VEC = [0.001]
BATCH_SIZE = 512
VALD_FRAC = 0.1

IMAGE_SHAPE = (50, 50, 1)
NUM_CLASSES = 3

CLASS_NAMES =['Undermelt', 'JustRight', 'Overmelt']

train_dir = '../../input/train_50/'
test_dir = '../../input/test_50/'
result_dir = 'Result_3/'

def network(input_shape = IMAGE_SHAPE, output_size = NUM_CLASSES, LR = 0.0001):
    '''
    Creates and returns CNN network
    '''
    
    inputLayer = Input(shape = input_shape)
    
    layer = BatchNormalization()(inputLayer)
    layer = Convolution2D(16, (3, 3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(layer)
    layer = BatchNormalization()(layer)
    #layer = Dropout(0.25)(layer)
    layer = Convolution2D(32, (3, 3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(layer)
    layer = BatchNormalization()(layer)
    #layer = Dropout(0.25)(layer)
    layer = Convolution2D(64, (3, 3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(layer)
    layer = BatchNormalization()(layer)
    #layer = Dropout(0.25)(layer)
    
    layer = Flatten()(layer)
    layer = Dense(256, activation='relu')(layer)
    #layer = BatchNormalization()(layer)
    layer = Dropout(0.25)(layer)
    layer = Dense(32, activation='relu')(layer)
    #layer = BatchNormalization()(layer)
    layer = Dropout(0.25)(layer)
    
    outputLayer = Dense(output_size, activation='softmax')(layer)
    
    model = Model(inputs=inputLayer, outputs=outputLayer)
    
    #p_model = multi_gpu_model(model, gpus=16)
    adam = Adam(lr = LR)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    return model

seed = 1
np.random.seed(seed) # for regenerating results

# Tensorflow GPU optimization
num_cores = 16
config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                        allow_soft_placement=True)
sess = tf.Session(config=config)
K.set_session(sess)

###--------- TRAINING AND VALIDATION ---------###

# read labels
with open(train_dir + 'labels.pkl', 'rb') as f:
    labels = pickle.load(f)

print(len(labels))
IDs = list(labels.keys())
np.random.shuffle(IDs)

# partition samples into training and validation
vald_start_index = int(len(IDs) * (1-VALD_FRAC))
partition = {'train':IDs[:vald_start_index], 'validation':IDs[vald_start_index:]}

# create batch data generators
training_generator = DataGenerator(train_dir,
                                   input_shape=IMAGE_SHAPE,
                                   output_size=NUM_CLASSES,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   augment=True).generate(labels,
                                                          partition['train'])
validation_generator = DataGenerator(train_dir,
                                     input_shape=IMAGE_SHAPE,
                                     output_size=NUM_CLASSES,
                                     batch_size=BATCH_SIZE,
                                     shuffle=False,
                                     augment=True).generate(labels,
                                                            partition['validation'])

# learn model for different learning rates
vald_accuracy = np.zeros_like(LR_VEC)
vald_loss = np.zeros_like(LR_VEC)
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
    
    vald_accuracy[i] = history.history['val_acc'][-1] # save final validation accuracy
    vald_loss[i] = history.history['val_loss'][-1] # save final validation loss

# print validation results
print('VALIDATION ACCURACY:')
print(vald_accuracy)
print('')
print('VALIDATION LOSS:')
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
                                  batch_size=BATCH_SIZE,
                                  shuffle=False,
                                  augment=False).generate(labels, IDs)

# evaluate the model on test dataset
test_score = model.evaluate_generator(testing_generator,
                                      steps = len(IDs)//BATCH_SIZE)

# print test results
print('TEST ACCURACY:')
print(test_score[1])
print('')
print('TEST LOSS:')
print(test_score[0])
print('')