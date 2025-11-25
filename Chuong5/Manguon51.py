from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Chia dữ liệu thành tập huấn luyện, phát triển và kiểm tra
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Huấn luyện trên tập huấn luyện
model.fit(X_train, y_train)

# Đánh giá trên tập phát triển
val_predictions = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, val_predictions))

# Nếu huấn luyện lại trên cả tập huấn luyện + phát triển
X_train_val = np.concatenate((X_train, X_val), axis=0)
y_train_val = np.concatenate((y_train, y_val), axis=0)
model.fit(X_train_val, y_train_val)

# Đánh giá cuối cùng trên tập kiểm tra với độ đo độ chính xác
test_predictions = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, test_predictions))
