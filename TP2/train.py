from sklearn import tree
from joblib import dump
import numpy as np

def train(X, Y, train):
    if(train):
        classifier = tree.DecisionTreeClassifier().fit(X, Y)

        # visualize
        tree.plot_tree(classifier)
        
        # save to file
        dump(classifier, 'classifier_model.joblib')
    else:
        classifier = load('classifier_model.joblib')

    return train

