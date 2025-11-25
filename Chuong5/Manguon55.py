from sklearn.metrics import silhouette_score 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 

# Tạo dữ liệu giả cho bài toán phân cụm 
X, y = make_blobs(n_samples=300, centers=4, random_state=42) 

# Áp dụng thuật toán KMeans để phân cụm 
kmeans = KMeans(n_clusters=4, random_state=42) 
y_pred = kmeans.fit_predict(X) 

# Tính hệ số Silhouette 
silhouette_avg = silhouette_score(X, y_pred) 
print(f"Hệ số Silhouette: {silhouette_avg}")
