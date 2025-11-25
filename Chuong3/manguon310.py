#Nhập thư viện
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
import numpy as np 
import matplotlib as mpl

#Thiết lập tham số để chuyển ảnh màu về ảnh đen trắng
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["image.cmap"] = "gray"

#Tải tập dữ liệu ảnh chân dung người
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

#Tạo một khung gồm 2 hàng mỗi hàng có 5 cột để hiển thị ảnh
print(‘Dữ liệu mẫu’)
fig, axes = plt.subplots(2, 5, figsize=(15, 8),
                         subplot_kw={'xticks': (), 'yticks': ()})
#Hiển thị 10 ảnh trong tập dữ liệu vào khung đã vẽ với tên người ở trên
for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])
plt.show()

#Thống kê số lượng ảnh, số lượng lớp
print("people.images.shape: {}".format(people.images.shape))
print("Số lớp: {}".format(len(people.target_names)))

#Liệt kê danh sách người và tên trong tập dữ liệu
counts = np.bincount(people.target)
for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print("{0:25} {1:3}".format(name, count), end='   ')
    if (i + 1) % 3 == 0:
        print()

#Tiền xử lý tập dữ liệu
mask = np.zeros(people.target.shape,bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1
    
X_people = people.data[mask]
y_people = people.target[mask]

#Phân tích PCA để giảm chiều dữ liệu
from sklearn.decomposition import PCA
pca = PCA(n_components=100, whiten=True, random_state=0)
X_pca = pca.fit_transform(X_people)

#Khởi tạo và phân cụm với thuật toán DBSCAN
from sklearn.cluster import DBSCAN
dbscan = DBSCAN()
labels = dbscan.fit_predict(X_pca)
print("Tham số ngầm định, số lượng loại nhãn: {}".format(np.unique(labels)))

#Thay đổi 
dbscan = DBSCAN(min_samples=3, eps=15)
labels = dbscan.fit_predict(X_pca)
print("Tham số min_samples=3, eps=15, số lượng nhãn: {}".format(np.unique(labels)))

#Đếm số phần tử trong các cụm và phần tử nhiễu
#Phần tử nhiễu được đánh số là -1 nên tính được 
#số phần tử trong 1 cụm ta cần +1 
#Cụm đầu tiên là các phần tử nhiễu
print("Số lượng phần tử/cụm: {}".format(np.bincount(labels + 1)))

#Phân tích một số phần tử được đánh số là nhiễu
noise = X_people[labels==-1]

fig, axes = plt.subplots(3, 9, subplot_kw={'xticks': (), 'yticks': ()},
                         figsize=(12, 4))
for image, ax in zip(noise, axes.ravel()):
    ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)  

#Thực nghiệm với giá trị khác nhau của tham số eps
for eps in [1, 3, 5, 7, 9, 11, 13]:
    print("\neps={}".format(eps))
    dbscan = DBSCAN(eps=eps, min_samples=3)
    labels = dbscan.fit_predict(X_pca)
    print("Số lượng cụm: {}".format(len(np.unique(labels))))
    print("Tổng số phần tử/cụm: {}".format(np.bincount(labels + 1)))

#Phân tích trường hợp tốt nhất
print(‘Phân tích trường hợp tốt nhất’)
dbscan = DBSCAN(min_samples=3, eps=7)
labels = dbscan.fit_predict(X_pca)

for cluster in range(max(labels) + 1):
    mask = labels == cluster
    n_images =  np.sum(mask)
    fig, axes = plt.subplots(1, n_images, figsize=(n_images * 1.5, 4),
                             subplot_kw={'xticks': (), 'yticks': ()})
    for image, label, ax in zip(X_people[mask], y_people[mask], axes):

        ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
        ax.set_title(people.target_names[label].split()[-1])
