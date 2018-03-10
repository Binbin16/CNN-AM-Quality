import numpy as np
import pickle

class DataGenerator():
    '''
    Generates data in batches for training
    '''
    def __init__(self, data_dir, input_shape = (300, 300, 3), output_size = 3, batch_size = 32, shuffle = True):
        'Initialization'
        self.data_dir = data_dir
        self.input_shape = input_shape
        self.output_size = output_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def generate(self, labels, list_IDs):
        'Generates batches of samples'
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
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(list_IDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)

            return indexes
        
    def __data_generation(self, labels, list_IDs_temp):
        'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        y = np.empty((self.batch_size, self.output_size), dtype = int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store volume
            with open(self.data_dir + str(ID) + '.pkl', 'rb') as f:
                X[i, :, :, :] = pickle.load(f)

            # Store class
            y[i] = labels[ID]

            return X, y