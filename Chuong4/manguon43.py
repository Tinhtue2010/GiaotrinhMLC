from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#Tải tập dữ liệu iris
df_iris = load_iris(as_frame=False)
X=df_iris.data
y=df_iris.target
#Chia tập dữ liệu thành tập huấn luyện và kiểm thử theo tỷ lệ 0.67:0.33
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=125
)
from sklearn.naive_bayes import GaussianNB
# Xây dựng bộ phân lớp Naive Bayes
model = GaussianNB()

#Huấn luyện mô hình
model.fit(X_train, y_train)

#Dự báo thử một phần tử dữ liệu 
predicted = model.predict([X_test[0]])
print("Actual Value:", y_test[0])
print("Predicted Value:", predicted[0])

#Tính toán các độ đo
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
y_pred = model.predict(X_test)
accuray = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average="weighted")

print("Accuracy:", accuray)
print("F1 Score:", f1)
