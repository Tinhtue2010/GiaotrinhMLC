from sklearn.preprocessing import Normalizer
import numpy as np

# Tạo dữ liệu mẫu
X = np.array([[1, -1, 2], [2, 3, -1]])
# Chuẩn hóa L1
scaler_l1 = Normalizer(norm='l1')

X_l1_normalized = scaler_l1.fit_transform(X)
print("Dữ liệu sau khi chuẩn hóa L1:")
print(X_l1_normalized)
