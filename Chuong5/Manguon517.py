from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Các mô hình cơ bản
estimators = [
    ('dt', DecisionTreeClassifier(max_depth=3)),
    ('svc', SVC(probability=True)),
    ('lr', LogisticRegression())
]

# Mô hình meta (mô hình tổng hợp)
meta_model = LogisticRegression()

# Tạo mô hình Stacking
stacking_model = StackingClassifier(estimators=estimators, final_estimator=meta_model)

# Huấn luyện mô hình
stacking_model.fit(X_train, y_train)

# Dự đoán và đánh giá mô hình
y_pred = stacking_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Stacking Model Accuracy: {accuracy:.4f}")
