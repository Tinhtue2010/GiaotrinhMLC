# Training data
df_num_train = df_train.select_dtypes('number')
df_num_train = df_num_train.drop(['Longitude','Latitude'],axis=1)
df_object_train = df_train.select_dtypes('object')
df_object_train = df_object_train.drop(['City','Zip Code', 'Customer ID', 'Churn Category', 'Churn Reason'],axis=1)

# Test data
df_num_test = df_test.select_dtypes('number')
df_num_test = df_num_test.drop(['Longitude','Latitude'],axis=1)
df_object_test = df_test.select_dtypes('object')
df_object_test = df_object_test.drop(['City','Zip Code', 'Customer ID', 'Churn Category', 'Churn Reason'],axis=1)
