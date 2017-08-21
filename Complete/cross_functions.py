import os, requests, zipfile, io
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier as cart, export_graphviz
import graphviz
import pydotplus


if not os.path.isdir('./UCI HAR Dataset/'):
    HAR_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
    req = requests.get(HAR_URL)
    compressed = zipfile.ZipFile(io.BytesIO(req.content))
    compressed.extractall()
    

def get_key_value_pairs(filename):
    '''Returns the key value pairs, separated by whitespaces,
    that are stored in a file.'''
    
    with open(filename, 'r') as file:
        for line in file.readlines():
            key, value = line.strip().split(' ')
            yield key, value

            
def print_accuracy(y_true, y_pred):
    print('Accuracy: {}'
      .format(np.round(accuracy_score(y_true=y_true, y_pred=y_pred, normalize=True), 4)))
    
    
def print_tree_graph(tree_model, features, activity, sel_feat):
    dot_data = export_graphviz(tree_model, out_file=None, 
                               feature_names=[features[el] for el in sel_feat],  
                               class_names=[activity[str(el)] for el in [1,2,3,4,5,6]]
                              )
    return pydotplus.graph_from_dot_data(dot_data)  


def compute_disjunctive_random_splits(x_sample, y_sample, splits):
    shuffled_dataset = x_sample.sample(frac=1, random_state=8)
    shuffled_splits =  np.array_split(shuffled_dataset, splits)
    masks = []
    for mask in range(0, splits):
        test_x = shuffled_splits[mask]
        train_x = x_sample[x_sample.index.isin(test_x.index) == False]
        test_y = y_sample.loc[shuffled_splits[mask].index]
        train_y = y_sample.loc[train_x.index]
        #
        masks.append([train_x, test_x, train_y, test_y])
    
    return masks
    
    

    
activity = {key: value for key, value in get_key_value_pairs('./UCI HAR Dataset/activity_labels.txt')}
features = [line.strip().split(' ')[1] for line in open('./UCI HAR Dataset/features.txt', 'r')]

Xtrain = pd.read_table('./UCI HAR Dataset/train/X_train.txt', header=None, delim_whitespace=True, names=features)
Xtrain['subject'] = [line.strip() for line in open('./UCI HAR Dataset/train/subject_train.txt','r')]
Xtrain['activity'] = pd.read_table('./UCI HAR Dataset/train/y_train.txt', header=None, delim_whitespace=True)
