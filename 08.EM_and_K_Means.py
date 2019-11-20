"""
8. Apply EM algorithm to cluster a set of data stored in a .CSV file. Use the same data set for
clustering using k-Means algorithm. Compare the results of these two algorithms and comment
on the quality of clustering. You can add Java/Python ML library classes/API in the program.
"""

import csv
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as sm
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

class_dict = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
l1 = [0, 1, 2]
with open('ds4.csv') as csvFile:
    dataset = [line for line in csv.reader(csvFile)]
    dataset = dataset[1:]
    X = []
    y = []
    for line in dataset:
        X.append(line[:-1])
        y.append(class_dict[line[-1]])

    X = np.array(X).astype(float)
    y = np.array(y).astype(int)


def rename_clusters(s):
    cnt = Counter((c1, c2) for c1, c2 in zip(s, y))
    most_common = cnt.most_common()
    map_dict = {}
    for tup in most_common:
        if not tup[0][0] in map_dict:
            map_dict[tup[0][0]] = tup[0][1]

    for i in range(len(s)):
        s[i] = map_dict[s[i]]
    return s


# EM part
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
y_kmeans = gmm.predict(X)
em = rename_clusters(y_kmeans)
plt.scatter(X[:, 0], X[:, 1], c=em, s=40, cmap='viridis')
print("Accuracy EM : ", sm.accuracy_score(y, em))
plt.show()

# K-means part
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
km = rename_clusters(y_kmeans)
plt.scatter(X[:, 0], X[:, 1], c=km, s=40, cmap='viridis')
print("Accuracy KM : ", sm.accuracy_score(y, km))
plt.show()

"""
Output:

Accuracy EM :  0.9666666666666667
Accuracy KM :  0.8933333333333333
"""
