
import h5py
import numpy as np

def to_categorical(y, num_classes=None):
    """
    Converts a class vector (integers) to binary class matrix.
    
    Args:
        y: class vector to be converted into a matrix
        num_classes: total number of classes
        
    Returns:
        A binary matrix representation of the input
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

class H5DataLoader(object):

    def __init__(self, data_path, is_train=True):
        self.is_train = is_train
        data_file = h5py.File(data_path, 'r')
        self.images, self.labels = data_file['X'], data_file['Y']
        self.gen_indexes()
    
    def gen_indexes(self):
        if self.is_train:
            self.indexes = np.random.permutation(range(self.images.shape[0]))
        else:
            self.indexes = np.array(range(self.images.shape[0]))
        self.cur_index = 0
    
    def next_batch(self, batch_size):
        next_index = self.cur_index+batch_size
        cur_indexes = list(self.indexes[self.cur_index:next_index])
        self.cur_index = next_index
        
        if len(cur_indexes) < batch_size and self.is_train:
            self.gen_indexes()
            self.cur_index = batch_size
            cur_indexes = list(self.indexes[:batch_size])
       
        if len(cur_indexes)==0 and not self.is_train:
            cur_indexes = [0]
        elif len(cur_indexes) < batch_size and not self.is_train:
            self.cur_index = 0
        cur_indexes.sort()
        
        outx, outy = self.images[cur_indexes], self.labels[cur_indexes]

        #print(len(np.unique(outy.flatten())))

        if 0: #len(np.unique(outy.flatten()))<3:
            outy = np.uint8(outy>128)
        else:
            outy = to_categorical(outy, num_classes=9)
        
        return outx, outy
