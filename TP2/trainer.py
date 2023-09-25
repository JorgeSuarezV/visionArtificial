import csv
import numpy as np
from sklearn import tree
from joblib import dump

# Step 1: Read the data from the CSV file
data = []
labels = []

with open('hu_moments.csv', 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    for row in csv_reader:
        labels.append(int(row[0]))
        hu_moments = [float(val) for val in row[1:]]
        data.append(hu_moments)

X = np.array(data)
Y = np.array(labels)

clasificador = tree.DecisionTreeClassifier().fit(X, Y)

# Step 4: Visualize the decision tree (optional)
# You can uncomment this section if you want to visualize the decision tree
tree.plot_tree(clasificador)

dump(clasificador, 'classifier.joblib')
