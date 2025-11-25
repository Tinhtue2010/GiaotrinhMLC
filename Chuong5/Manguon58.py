from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Tải dữ liệu và chia thành tập huấn luyện và kiểm tra
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

# Định nghĩa mô hình và các tham số cần tinh chỉnh
model = RandomForestClassifier()
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Khởi tạo GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

# Huấn luyện mô hình với Grid Search
grid_search.fit(X_train, y_train)

# In kết quả
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
