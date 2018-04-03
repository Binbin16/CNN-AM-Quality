import numpy as np
import pickle
from PIL import Image

in_train_dir = '../Data/input/train/'
in_test_dir = '../Data/input/test/'
out_train_dir = '../Data/input/train_50/'
out_test_dir = '../Data/input/test_50/'

image_shape_in = (300, 300, 3)
image_shape_out = (50, 50, 1)

###--------- TRAINING AND VALIDATION DATA ---------###

# Copy labels file
with open(in_train_dir + 'labels.pkl', 'rb') as f:
    labels = pickle.load(f)

with open(out_train_dir + 'labels.pkl', 'wb') as f:
    pickle.dump(labels, f)

for i, ID in enumerate(labels.keys()):
    with open(in_train_dir + str(ID) + '.pkl', 'rb') as f:
        im = pickle.load(f)
        im = Image.fromarray(np.uint8(im*255), 'RGB').convert('L').resize((image_shape_out[0], image_shape_out[1]), Image.ANTIALIAS)
        im = np.array(im)/255
    with open(out_train_dir + str(ID) + '.pkl', 'wb') as f:
        pickle.dump(im, f)
        
###--------- TESTING DATA ---------###

# Copy labels file
with open(in_test_dir + 'labels.pkl', 'rb') as f:
    labels = pickle.load(f)

with open(out_test_dir + 'labels.pkl', 'wb') as f:
    pickle.dump(labels, f)

for i, ID in enumerate(labels.keys()):
    with open(in_test_dir + str(ID) + '.pkl', 'rb') as f:
        im = pickle.load(f)
        im = Image.fromarray(np.uint8(im*255), 'RGB').convert('L').resize((image_shape_out[0], image_shape_out[1]), Image.ANTIALIAS)
        im = np.array(im)/255
    with open(out_test_dir + str(ID) + '.pkl', 'wb') as f:
        pickle.dump(im, f)
