from sklearn.ensemble import GradientBoostingClassifier

# Tạo mô hình Gradient Boosting 
gdbt_model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, random_state=42)
# Huấn luyện mô hình
gdbt_model.fit(X_train, y_train)

# Dự đoán và đánh giá mô hình
y_pred = gdbt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Gradient Boosting Model Accuracy: {accuracy:.4f}")
