from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from scipy.stats import randint

# Tải dữ liệu và chia thành tập huấn luyện và kiểm tra
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

# Định nghĩa mô hình và các tham số cần tinh chỉnh
model = RandomForestClassifier()
param_dist = {
    'n_estimators': randint(10, 200),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 20)
}

# Khởi tạo RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100, cv=5, scoring='accuracy', random_state=42)

# Huấn luyện mô hình với Random Search
random_search.fit(X_train, y_train)

# In kết quả
print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)
