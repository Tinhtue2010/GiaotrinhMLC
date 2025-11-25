encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
encoder.fit(df_object_train[df_object_train.columns])
Khi handle_unknown được thiết lập là 'use_encoded_value', OrdinalEncoder sẽ sử dụng giá trị unknown_value (trong trường hợp này là -1) để mã hóa các giá trị phân loại mà nó không gặp trong dữ liệu huấn luyện. 
Hàm sau đây có thể được sử dụng để hiển thị các danh mục hiện có:
encoder.categories_
[array(['Female', 'Male'], dtype=object),
 array(['No', 'Yes'], dtype=object),
 array(['None', 'Offer A', 'Offer B', 'Offer C', 'Offer D', 'Offer E'],
       dtype=object),
 array(['No', 'Yes'], dtype=object),
 array(['No', 'Yes'], dtype=object),
 array(['No', 'Yes'], dtype=object),
 array(['Cable', 'DSL', 'Fiber Optic', 'None'], dtype=object),
 array(['No', 'Yes'], dtype=object),
 array(['No', 'Yes'], dtype=object),
 array(['No', 'Yes'], dtype=object),
 array(['No', 'Yes'], dtype=object),
 array(['No', 'Yes'], dtype=object),
 array(['No', 'Yes'], dtype=object),
 array(['No', 'Yes'], dtype=object),
 array(['No', 'Yes'], dtype=object),
 array(['Month-to-Month', 'One Year', 'Two Year'], dtype=object),
 array(['No', 'Yes'], dtype=object),
 array(['Bank Withdrawal', 'Credit Card', 'Mailed Check'], dtype=object),
 array(['Churned', 'Stayed'], dtype=object),
 array(['18-25', '25-35', '35-45', '45-60', '60-80'], dtype=object)]
