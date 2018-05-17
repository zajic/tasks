import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv", delimiter=",", header = 0)

#rename column names  from '"x"' to 'x'
old_col_names = list(data.columns)
new_col_names = [name.replace('"',"").strip() for name in old_col_names]
data = data.rename(index=str, columns = dict(zip(old_col_names, new_col_names)))

X = data[['x','y']]

#fit model - I chose eps and min_samples more or less arbitrarily here to get 2 clusters
db = DBSCAN(eps=25, min_samples = 9).fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: {}'.format(n_clusters_))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

color_map = dict(zip(unique_labels, colors))
color_labels = [color_map[label] if label != -1 else (0,0,0,1) for label in labels]

plt.scatter(X.x, X.y, c=color_labels)
plt.title('Estimated number of clusters: {}'.format(n_clusters_))
plt.show()