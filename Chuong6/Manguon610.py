# Tính các độ đo đánh giá mô hình
def evaluate(y_test, pred):
    acc = accuracy_score(y_test, pred)
    print("ACC: ", acc)
    print(classification_report(y_test,pred))

# Trực quan hoá một số thông tin
def visualize_result(y_test, pred):
    sns.heatmap(confusion_matrix(y_test,pred),cmap='Blues',annot=True, fmt="d",linewidths=2,linecolor='white')

# In kết quả
print(evaluate(y_test, y_pred))
visualize_result(y_test, y_pred)
