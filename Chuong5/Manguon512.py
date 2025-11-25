import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# Tạo dữ liệu giả lập
X = np.random.rand(1000, 20)  # 1000 mẫu, mỗi mẫu có 20 đặc trưng
y = np.random.randint(0, 2, size=(1000,))  # 1000 nhãn nhị phân (0 hoặc 1)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình học máy (ví dụ: Logistic Regression)
model = LogisticRegression(max_iter=1, warm_start=True, solver='lbfgs')  # max_iter=1 để huấn luyện 1 bước mỗi lần

# Cài đặt Early Stopping
patience = 5  # Số epoch không cải thiện mất mát trên tập kiểm tra trước khi dừng
best_loss = float('inf')  # Mất mát tốt nhất, bắt đầu với vô cùng lớn
epochs_without_improvement = 0  # Đếm số epoch không cải thiện

for epoch in range(100):  # Số lượng epoch tối đa
    # Huấn luyện mô hình trong 1 bước
    model.fit(X_train, y_train)
    
    # Dự đoán và tính toán mất mát trên tập kiểm tra
    y_pred = model.predict_proba(X_test)[:, 1]  # Dự đoán xác suất cho lớp 1
    current_loss = log_loss(y_test, y_pred)
    
    print(f'Epoch {epoch+1}, Loss: {current_loss}')
    
    # Kiểm tra có cải thiện mất mát không
    if current_loss < best_loss:
        best_loss = current_loss
        epochs_without_improvement = 0  # Reset nếu có cải thiện
    else:
        epochs_without_improvement += 1
    
    # Nếu không có cải thiện trong "patience" epoch, dừng huấn luyện
    if epochs_without_improvement >= patience:
        print(f'Early stopping at epoch {epoch+1} with loss {current_loss}')
        break

print("Huấn luyện kết thúc!")
