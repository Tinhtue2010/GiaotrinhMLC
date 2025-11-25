from sklearn.metrics import mean_squared_error
# Giả sử y_true là giá trị thực tế và y_pred là giá trị dự đoán 
# Tính MSE 
mse = mean_squared_error(y_true, y_pred) 

# Tính RMSE 
rmse = np.sqrt(mse)mae = mean_squared_error (y_true, y_pred) 
print(f'Mean Squared Error (MSE): {mse}')
