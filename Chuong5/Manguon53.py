from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression 
from sklearn.datasets import load_iris 
# Tải dữ liệu iris 
data = load_iris() 
X = data.data 
y = data.target 

# Khởi tạo mô hình Logistic Regression 
model = LogisticRegression(max_iter=200) 

# Thực hiện kiểm thử chéo và lấy thêm thông tin về thời gian huấn luyện và kiểm tra 
results = cross_validate(model, X, y, cv=5, scoring='accuracy', return_train_score=True) 

print("Test scores for each fold:", results['test_score']) 
print("Train scores for each fold:", results['train_score']) 
print("Fit time for each fold:", results['fit_time']) 
print("Score time for each fold:", results['score_time']) 
print("Average test score:", results['test_score'].mean())
