import matplotlib.pyplot as plt
import numpy as np


def normalize(array):
    """ Rescale images values to 0 to 1
    """
    min_ = np.min(array)
    max_ = np.max(array)
    return (array - min_)/(max_ - min_)

def hot_encode(N, array):
    """ For each element in the array tranlate a number
        category into a vector with N elements whose i'th
        element is 1. The 0'th element is always 0
        Example:
        [1, 2, 3] -- N=4 --> [[0,1,0,0], [0,0,1,0], [0,0,0,1]]
    """
    array_length = len(array)
    encoded_array = np.ndarray((array_length, N))
    encoded_array.fill(0)
    for i in range(array_length):
        encoded_array[i][array[i][0]] = 1
    return encoded_array

def create_shuffle_mask(n):
    """ Create an array with with elements from 0 to n
        in random order.
        Example:
        n = 4 -> [2, 1, 0, 3]
    """
    mask = list(range(n))
    np.random.shuffle(mask)
    return mask

def split_array(array, fraction, first):
    """ Return a fraction of an array, first define if
        should return the first or the last part.
        Example:
        [1,2,3,4] fraction=0.5 first=False -> [3,4]
    """
    shape = list(array.shape)
    shape[0] = int(shape[0] * fraction)
    shape = tuple(shape)
    
    split = np.ndarray(shape)
    if first:
        split = array[:shape[0]]
    else:
        split = array[shape[0]:]
    
    return split

def split_train_test(array, train_fraction = 0.7):
    """ Divide an array into two arrays (train, test)
        train array will contain train_fraction of
        the elements of array.
    """
    return (split_array(array, train_fraction, True),
            split_array(array, train_fraction, False))

def shuffle(array, mask):
    """ Given an array and a ordering mask, create a new
        array with elemets ordered by the mask.
        Example:
        array=[a, b, c, d] mask=[0, 1, 3, 2] -> [a, b, d, c]
    """
    shuffle_array = np.ndarray(array.shape, dtype=float)
    for i in range(len(array)):
        shuffle_array[i] = array[mask[i]]
    return shuffle_array

def make_batched_dataset(array, batch_size):
    """ Divede array in subarrays of batches of batch_size elements.
    """
    array_length, array_width = array.shape
    num_batches = array_length//batch_size
    assert(array_length % batch_size == 0)

    batched_array = np.ndarray((num_batches, batch_size, array_width))
    for i, offset in enumerate(range(0, array_length, batch_size)):
        batched_array[i] = array[offset:offset+batch_size]
        
    return batched_array

def print_mnist(ix, mnist, label, predicted=None, ax=None):
    """ Print the image in the index ix, from the mnist dataset 
        and show the corresponding label and prediction(if any).
    """
    if not ax:
        ax = plt.subplot(111)
    ax.matshow(mnist[ix].reshape((28,28)))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if type(predicted) != type(None):
        ax.set_title('predicted = {}, label = {}'.format(
                np.argmax(predicted), np.argmax(label[ix])))    
    else:
        ax.set_title('label = {}'.format(np.argmax(label[ix])))
        
def show_errors(mnist, label, errors, predictions):
    """ Create a 4x4 grid with the erroneous predictions (maximum of 20)
    """
    max_prints = 20
    to_print = min(sum([ not e for e in errors]), max_prints)
    cols = 4
    rows = to_print//cols
    if to_print % cols > 0:
        rows += 1
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols*3,rows*3))
    prints = 0
    for i, is_error in enumerate(errors):
        if not is_error:
            print_mnist(i, mnist, label, predictions[i], 
                        axes[prints//cols, prints%cols])
            prints += 1
            if prints >= to_print:
                break