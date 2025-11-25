from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris 

# Tải dữ liệu iris 
data = load_iris() 
X = data.data 
y = data.target 

# Khởi tạo mô hình Logistic Regression 
model = LogisticRegression(max_iter=200) 

# Thực hiện kiểm thử chéo và tính điểm accuracy 
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy') 
print("Accuracy scores for each fold:", scores)
print("Average accuracy:", scores.mean())
