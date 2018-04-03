import numpy as np
import pickle
from PIL import Image

class DataGenerator():
    '''
    Generates data in batches for training
    '''
    def __init__(self, data_dir, input_shape = (300, 300, 3), output_size = 3, batch_size = 32, shuffle = True, augment = True):
        '''
        Initialization
        '''
        self.data_dir = data_dir
        self.input_shape = input_shape
        self.output_size = output_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        
    def generate(self, labels, list_IDs):
        '''
        Generates batches of samples
        '''
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_IDs)

            # Generate batches
            imax = len(indexes) // self.batch_size
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

                # Generate data
                X, y = self.__data_generation(labels, list_IDs_temp)

                yield X, y
                
    def __get_exploration_order(self, list_IDs):
        '''
        Generates order of exploration
        '''
        # Find exploration order
        indexes = np.arange(len(list_IDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)

        return indexes
        
    def __data_generation(self, labels, list_IDs_temp):
        '''
        Generates data of batch_size samples
        '''
        # Initialization
        X = np.empty((self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        y = np.empty((self.batch_size, self.output_size), dtype = int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store volume
            with open(self.data_dir + str(ID) + '.pkl', 'rb') as f:
                im = pickle.load(f)
                
            if (self.input_shape[2] == 1):
                X[i, :, :, 0] = im # for input size 50x50x1
            else:
                X[i, :, :, :] = im # for input size 300x300x3
                
            # Store class
            y[i] = labels[ID]
        
        if self.augment: # randomly mirror the images vertically and horizontally
            if (np.random.random() > 0.5):
                X = np.flip(X, axis = 1) # flip vertically
            if (np.random.random() > 0.5):
                X = np.flip(X, axis = 2) # flip horizontally
        return X, y
