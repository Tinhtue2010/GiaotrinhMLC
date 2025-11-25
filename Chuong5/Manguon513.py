from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Tải dữ liệu mẫu
data = load_iris()
X = data.data
y = data.target

# Chia dữ liệu thành bộ huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Tạo mô hình Bagging với cây quyết định làm mô hình cơ bản
base_model = DecisionTreeClassifier()
bagging_model = BaggingClassifier(base_estimator=base_model, n_estimators=50, random_state=42)

# Huấn luyện mô hình
bagging_model.fit(X_train, y_train)

# Dự đoán và đánh giá mô hình
y_pred = bagging_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Bagging Model Accuracy: {accuracy:.4f}")
