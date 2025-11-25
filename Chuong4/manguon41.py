from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from numpy import sqrt
import pandas as pd
import numpy as np

#Tải dữ liệu từ trang web về, trong đó bỏ đi 22 dòng đầu tiên không phải dữ liệu
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
#Vì dòng thông tin dài quá nên bị cắt thành 2 dòng, 
# tiến hành ghép và tách phần dữ liệu và nhãn ra
#Tách các cột đặc trưng và cột đích vào 2 biến x và y
x = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]

#Chia tập dữ liệu thành tập huấn luyện và kiểm thử
xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.15)

#Khởi tạo bộ hồi quy
bay_ridge = BayesianRidge()
#print(bay_ridge)
BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True,
       fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=300,
       tol=0.001, verbose=False) 

#Huấn luyện mô hình 
bay_ridge.fit(xtrain, ytrain)

#Kiểm thử mô hình
score=bay_ridge.score(xtrain, ytrain)

#In kết quả hiển thị
print("Model score (R-squared): %.2f" % score)
ypred = bay_ridge.predict(xtest)
mse = mean_squared_error(ytest, ypred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % sqrt(mse))

#Vẽ biểu đồ mô tả kết quả
x_ax = range(len(ytest))
plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()
