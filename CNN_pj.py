import numpy as np

from keras.models import model_from_json, load_model
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers import Convolution2D, merge, MaxPooling2D, Input, Add, Concatenate
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical

import tensorflow as tf

# PARAMETERS
EPOCHS = 1 # Number of iterations on the dataset
LR_VEC = [0.0001] #
BATCH_SIZE = 16

IMAGE_SHAPE = (300, 300, 3)
NUM_CLASSES = 3

CLASS_NAMES =['Undermelt', 'JustRight', 'Overmelt']

data_dir = '../../Data/'
result_dir = 'Result/'

num_batches = 5
NFOLDS = 5 # same as num_batches

train_files = [data_dir + 'data_batch_' + str(ii) + '.bin' for ii in range(1, num_batches+1)]
test_file = data_dir + 'test_batch.bin'

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

def get_batch(filepath, image_shape = IMAGE_SHAPE):
    '''
    Reads and returns a batch of images and labels from file 'filepath'
    '''
    
    label_bytes = 1
    image_bytes = np.prod(image_shape)
    row_bytes = label_bytes + image_bytes
    
    f = open(filepath, 'rb')
    data = np.fromfile(f, dtype=np.uint8)
    
    batch_size = data.size // row_bytes
    data = np.reshape(data, (batch_size, -1))
    
    labels = to_categorical(data[:, 0])
    images = np.transpose(np.reshape(data[:, 1:], [batch_size, image_shape[2], image_shape[0], image_shape[1]]), (0, 2, 3, 1)) / 255
    
    return images, labels

#Tensorflow GPU optimization
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)

vald_accuracy = np.zeros_like(LR_VEC)
for i, lr in enumerate(LR_VEC):
    
    print('**************************************')
    print('Running with learning rate ' + str(lr))
    
    # using 5-folds cross-validation
    for fold in range(NFOLDS):
        
        print(' --- Fold # ' + str(fold))
        model = network(LR = lr)
        
        images = np.array([])
        labels = np.array([])
        for batch, filepath in enumerate(train_files):
            if (batch == fold): # skip one batch for validation
                continue
            img, lbl = get_batch(filepath)
            
            if (labels.size == 0):
                images = img
                labels = lbl
            else:
                images = np.append(images, img, axis=0)
                labels = np.append(labels, lbl, axis=0)
        print(' ------ Data prepared. Training...')
        model.fit(images, labels, batch_size=BATCH_SIZE, epochs=EPOCHS)
        
        vald_file = train_files[fold]
        images, labels = get_batch(vald_file)
        score = model.evaluate(images, labels)
        print(' --- Validation accuracy on fold # ' + str(fold) + ' = ' + str(score[1]))
        vald_accuracy[i] += score[1]
    vald_accuracy[i] /= NFOLDS
    print('')
    print('Validation accuracy = ' + str(vald_accuracy[i]))
    print('')

print('**************************************')

# find LR that gives maximum cross-validation accuracy
mx_i = np.argmax(vald_accuracy)
lr = LR_VEC[mx_i]
print('Maximum validation accuracy ' + str(vald_accuracy[mx_i]) + ' with LR = ' + str(lr))

# Train final model with the chosen LR
print('Training final model with learning rate ' + str(lr))
model = network(LR = lr)
images = np.array([])
labels = np.array([])
for batch, filepath in enumerate(train_files):
    img, lbl = get_batch(filepath)
    if (labels.size == 0):
        images = img
        labels = lbl
    else:
        images = np.append(images, img, axis=0)
        labels = np.append(labels, lbl, axis=0)

print(' ------ Data prepared. Training...')
model.fit(images, labels, batch_size=BATCH_SIZE, epochs=EPOCHS)

images, label = get_batch(test_file)
score = model.evaluate(images, labels)
print(' --- Test loss = ' + str(score[1]))
print(' --- Test accuracy = ' + str(score[1]))
model.save(result_dir + 'keras_model_lr_' + str(lr) + '.h5')
print('**************************************')