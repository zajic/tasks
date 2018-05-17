from sklearn.cluster import KMeans
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

data = pd.read_csv("data.csv", delimiter=",", header = 0)

#rename column names  from '"x"' to 'x'
old_col_names = list(data.columns)
new_col_names = [name.replace('"',"").strip() for name in old_col_names]
data = data.rename(index=str, columns = dict(zip(old_col_names, new_col_names)))

#fit model
X = data[['x','y']]
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
data.ClusterID = kmeans.labels_
print("Inertia: {:.{}f}".format(kmeans.inertia_,2))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, kmeans.labels_))

# assign different colours to clusters
COLOR_MAP = {0: 'r',
            1: 'b'}

color_labels = [COLOR_MAP[cluster_id] for cluster_id in data.ClusterID]

#plot clusters
plt.scatter(data.x, data.y, c=color_labels)

#plot centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c = 'k', marker = 'v', s=30)
centroid_marker = mlines.Line2D([], [], color='black', marker='v', linestyle='None',
                          markersize=10, label='Centroids')
plt.legend(handles=[centroid_marker])

plt.show()