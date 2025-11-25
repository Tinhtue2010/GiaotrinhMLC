path = '[đường dẫn đến thư mục chứa file'
filename = 'telecom_customer_churn.csv'
print(f'Load data from {filename}.')
df = pd.read_csv(path+filename)
