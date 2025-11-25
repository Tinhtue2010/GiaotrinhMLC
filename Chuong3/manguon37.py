from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import mglearn
#Tạo tập dữ liệu giả lập
X, y = make_blobs(random_state=1)
agg = AgglomerativeClustering(n_clusters=3)
model= agg.fit(X)
assignment = model.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=assignment, cmap=mglearn.cm3, s=60)
print(X[1])
