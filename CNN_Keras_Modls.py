import numpy as np
import pickle
import datetime

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
Model_ID = '22'
EPOCHS = 200 # Number of iterations on the dataset
LR_VEC = [0.001] #
BATCH_SIZE = 512
VALD_FRAC = 0.1
num_cores = 12
Test_ID = '03L'

print('**PARAMETERS**********')
print('Model: \t\t', Model_ID)
print('EPOCHS:\t\t', EPOCHS)
print('LR: \t\t', LR_VEC)
print('Batch size:\t', BATCH_SIZE)
print('VALD_FRAC:\t', VALD_FRAC)
print('Num cores:\t', num_cores,'\n')

print('BN: \t\t YYN')
print('L2 regul:\t Y0.1Y0.1')
print('Dropout:\t Y0.5Y0.5')
print('Conv1 filter:\t 3x3/16, 3x3/32, 256,32')

print('Test_ID:\t\t', Test_ID)



IMAGE_SHAPE = (50, 50, 1)
NUM_CLASSES = 3
print('Image size: \t\t', IMAGE_SHAPE)

CLASS_NAMES =['Undermelt', 'JustRight', 'Overmelt']



train_dir = '../../input/train_50/'
test_dir = '../../input/test_50/'
result_dir = 'Result/Result_'+ Model_ID + '/'


#train_files = [data_dir + 'data_batch_' + str(ii) + '.bin' for ii in range(1, num_batches+1)]
#test_file = data_dir + 'test_batch.bin'

def network(input_shape = IMAGE_SHAPE, output_size = NUM_CLASSES, LR = 0.0001):
    '''
    Creates and returns CNN network
    '''
    
    inputLayer = Input(shape = input_shape)
    
    layer = BatchNormalization()(inputLayer)
    
    layer = Convolution2D(16, (3, 3), padding='same', activation='relu',kernel_regularizer=l2(0.1))(layer) #kernel_regularizer=l2(0.01)
    layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.5)(layer)
    
    layer = Convolution2D(32, (3, 3), padding='same', activation='relu',kernel_regularizer=l2(0.1))(layer)
    layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.5)(layer)
    
    #layer = Convolution2D(64, (3, 3), padding='same', activation='relu')(layer)
    #layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(layer)
    #layer = BatchNormalization()(layer)
    #layer = Dropout(0.25)(layer)
    
    layer = Flatten()(layer)
    layer = Dense(256, activation='relu',kernel_regularizer=l2(0.1))(layer)  
    layer = Dropout(0.5)(layer)
    layer = Dense(32, activation='relu',kernel_regularizer=l2(0.1))(layer)
    layer = Dropout(0.5)(layer)
    
    outputLayer = Dense(output_size, activation='softmax')(layer)
    
    model = Model(inputs=inputLayer, outputs=outputLayer)
    
    adam = Adam(lr = LR)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    return model

seed = 0
np.random.seed(seed) # for regenerating results
tf.set_random_seed(seed)

# TODO: Tensorflow GPU optimization
#num_cores = 128
#GPU=False
#CPU=True
#if GPU:
#    num_GPU = 2
#    num_CPU = 1
#if CPU:
#    num_CPU = 16
#    num_GPU = 0
#
#config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
#        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
#        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
#sess = tf.Session(config=config)
#K.set_session(sess)

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                        allow_soft_placement=True)
sess = tf.Session(config=config)
K.set_session(sess)

###--------- TRAINING AND VALIDATION ---------###
starttime1 = datetime.datetime.now()
# read labels
with open(train_dir + 'labels.pkl', 'rb') as f:
    labels = pickle.load(f)

IDs = list(labels.keys())
np.random.shuffle(IDs)

# partition samples into training and validation
vald_start_index = int(len(IDs) * (1-VALD_FRAC))
partition = {'train':IDs[:vald_start_index], 'validation':IDs[vald_start_index:]}

print('Train data size: \t\t', len(partition['train']))
print('Validation data size:\t ', len(partition['validation']))

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
    
    
    print('Running with learning rate ' + str(lr))
    
    model = network(LR = lr)
    history = model.fit_generator(generator = training_generator,
                                  steps_per_epoch = len(partition['train'])//BATCH_SIZE,
                                  validation_data = validation_generator,
                                  validation_steps = len(partition['validation'])//BATCH_SIZE,
                                  epochs = EPOCHS,
                                  verbose = 0)
    
    # save model and results
    model.save( result_dir + 'keras_Model-'+ Model_ID + '_T-'+ Test_ID  +'_' + str(i) + '.h5')
    print('Result Directory: \t', result_dir + 'keras_Model-'+ Model_ID + '_T-'+ Test_ID  +'_' + str(i) + '.h5')
    print('Result Directory: \t', result_dir + 'keras_Model-'+ Model_ID + '_T-'+ Test_ID  +'_' + str(i) + '_history.pkl')
    with open(result_dir + 'keras_Model-'+ Model_ID + '_T-'+ Test_ID  +'_' + str(i) + '_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    
    #print(history.history.keys())
    #print(history.history.values())
    vald_accuracy[i] = history.history['val_acc'][-1] # save final validation accuracy

print('**RESULTS***************')
print('VALIDATION ACCURACY:\t',vald_accuracy ,'\n')

endtime1 = datetime.datetime.now()

###--------- TESTING ---------###
starttime2 = datetime.datetime.now()

# find the best LR and load corresponding model : maximum validation accuracy
mx_i = np.argmax(vald_accuracy)
model = load_model( result_dir + 'keras_Model-'+ Model_ID + '_T-'+ Test_ID  +'_' + str(i) + '.h5')

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
print('TEST SCORE:\t\t',test_score ,'\n')

endtime2 = datetime.datetime.now()
print('Train and Val time:\t ', endtime1-starttime1)
print('Testing time:\t\t ', endtime2-starttime2)
