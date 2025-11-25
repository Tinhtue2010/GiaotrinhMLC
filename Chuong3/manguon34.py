from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#Tạo dữ liệu giả lập 2 chiều
X, y = make_blobs(random_state=1)
#Hiển thị dữ liệu
print(X)
#Xây dựng mô hình phân cụm với số cụm là 3
kmeans =KMeans(copy_x=True, init='k-means++', max_iter=300, n_clusters=3, n_init=10, random_state=None, tol=0.0001,verbose=0)
kmeans.fit(X)
#In nhãn cụm
print(kmeans.labels_)

#Gán tên cụm cho dữ liệu để vẽ
assignment = kmeans.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=assignment, cmap=mglearn.cm3, s=60)

#Dự đoán nhãn của dữ liệu mới
T=[[-8,-4]]
print(T)
print(kmeans.predict(T))
