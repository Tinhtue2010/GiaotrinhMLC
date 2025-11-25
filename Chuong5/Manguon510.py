from hyperopt import fmin, tpe, hp, Trials
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Tải dữ liệu và chia thành tập huấn luyện và kiểm tra
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

# Định nghĩa hàm mục tiêu
def objective(params):
    model = RandomForestClassifier(n_estimators=params['n_estimators'], 
                                   max_depth=params['max_depth'], 
                                   min_samples_split=params['min_samples_split'])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return -accuracy  # Chúng ta tối ưu hóa ngược lại (minimize negative accuracy)

# Định nghĩa không gian tham số
space = {
    'n_estimators': hp.choice('n_estimators', [10, 50, 100, 150, 200]),
    'max_depth': hp.choice('max_depth', [None, 10, 20, 30]),
    'min_samples_split': hp.choice('min_samples_split', [2, 5, 10, 15])
}

# Khởi tạo Trials (giám sát tiến trình tối ưu hóa)
trials = Trials()

# Tiến hành tối ưu hóa với thuật toán Tree of Parzen Estimators (TPE)
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

# In kết quả
print("Best Parameters:", best)
