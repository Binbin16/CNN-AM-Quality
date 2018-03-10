import numpy as np
import pickle
from keras.utils import to_categorical

data_dir = '../../Data/'
out_train_dir = '../../input/train/'
out_test_dir = '../../input/test/'

num_batches = 5

train_files = [data_dir + 'data_batch_' + str(ii) + '.bin' for ii in range(1, num_batches+1)]
test_file = data_dir + 'test_batch.bin'

image_shape = (300, 300, 3)

label_bytes = 1
image_bytes = np.prod(image_shape)
row_bytes = label_bytes + image_bytes

# PREPARE TRAIN DATASET
ID = 0
labels = {}
for filepath in train_files:
    f = open(filepath, 'rb')
    data = np.fromfile(f, dtype=np.uint8)
    
    batch_size = data.size // row_bytes
    data = np.reshape(data, (batch_size, -1))
    
    y = to_categorical(data[:, 0])
    X = np.transpose(np.reshape(data[:, 1:], [batch_size, image_shape[2], image_shape[0], image_shape[1]]), (0, 2, 3, 1)) / 255
    
    keys = range(ID, ID+batch_size) # create unique IDs as keys
    ID += batch_size
    
    labels.update(dict(zip(keys, y))) # insert labels with ID keys
    # pickle each train image as <ID>.pkl
    for i in range(batch_size):
        file = out_train_dir + str(keys[i]) + '.pkl'
        with open(file, 'wb') as f:
            pickle.dump(X[i, :, :, :], f)
            
# pickle train labels
file = out_train_dir + 'labels.pkl'
with open(file, 'wb') as f:
    pickle.dump(labels, f)
    
# PREPARE TEST DATASET
# continued ID from train dataset
labels = {}

f = open(test_file, 'rb')
data = np.fromfile(f, dtype=np.uint8)

batch_size = data.size // row_bytes
data = np.reshape(data, (batch_size, -1))

y = to_categorical(data[:, 0])
X = np.transpose(np.reshape(data[:, 1:], [batch_size, image_shape[2], image_shape[0], image_shape[1]]), (0, 2, 3, 1)) / 255

keys = range(ID, ID+batch_size) # create unique IDs as keys
ID += batch_size

labels.update(dict(zip(keys, y))) # insert labels with ID keys
# pickle each test image as <ID>.pkl
for i in range(batch_size):
    file = out_test_dir + str(keys[i]) + '.pkl'
    with open(file, 'wb') as f:
        pickle.dump(X[i, :, :, :], f)

# pickle train labels
file = out_test_dir + 'labels.pkl'
with open(file, 'wb') as f:
    pickle.dump(labels, f)