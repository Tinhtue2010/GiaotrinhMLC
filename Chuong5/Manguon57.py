from sklearn.preprocessing import Normalizer
import numpy as np

# Tạo dữ liệu mẫu
X = np.array([[1, -1, 2], [2, 3, -1]])

# Chuẩn hóa L2
scaler_l2 = Normalizer(norm='l2')
X_l2_normalized = scaler_l2.fit_transform(X)

print("Dữ liệu sau khi chuẩn hóa L2:")
print(X_l2_normalized)
