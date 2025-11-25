from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
# Tạo mô hình AdaBoost 

ada_boost_model = AdaBoostClassifier(n_estimators=50, learning_rate=0.05, random_state=42)
# Huấn luyện mô hình
ada_boost_model.fit(X_train, y_train)

# Dự đoán và đánh giá mô hình
y_pred = ada_boost_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"AdaBoost Model Accuracy: {accuracy:.4f}")
