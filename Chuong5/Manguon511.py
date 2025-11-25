import numpy as np

# Hàm huấn luyện Mini-batch Gradient Descent
def mini_batch_gradient_descent(X, y, model, batch_size, learning_rate, epochs):
    # X: dữ liệu đầu vào, y: nhãn, model: lớp mô hình với các phương thức dự đoán và tính toán mất mát
    n = len(X)  # Số lượng mẫu trong dữ liệu
    for epoch in range(epochs):  # Lặp qua số epoch (số lần huấn luyện)
        # Xáo trộn dữ liệu
        indices = np.random.permutation(n)
        X_shuffled, y_shuffled = X[indices], y[indices]
        
        # Lặp qua các mini-batch
        for i in range(0, n, batch_size):  # Lặp qua các batch với kích thước mini-batch
            # Tạo mini-batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Tính toán gradient và cập nhật tham số mô hình
            gradients = model.compute_gradients(X_batch, y_batch)  # Tính gradient cho mini-batch
            model.update_parameters(gradients, learning_rate)  # Cập nhật tham số với gradient và learning rate
        
        # In ra giá trị mất mát (loss) mỗi 10 epoch để theo dõi quá trình huấn luyện
        if epoch % 10 == 0:
            loss = model.compute_loss(X, y)  # Tính toán mất mát trên toàn bộ dữ liệu
            print(f"Epoch {epoch}, Loss: {loss}")
