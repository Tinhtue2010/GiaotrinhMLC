import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
df = pd.read_csv("./GroceryStoreDataSet.csv", names = ['transaction'], sep = ',')
df = list(df["transaction"].apply(lambda x:x.split(",")))
df
one_hot_transformer = TransactionEncoder()
df_transform = one_hot_transformer.fit_transform(df)
df = pd.DataFrame(df_transform,columns=one_hot_transformer.columns_)
df
from mlxtend.frequent_patterns import fpgrowth

#Gợi giải thuật FP-growth với ngưỡng support là 0.2
res=fpgrowth(df, min_support=0.2, use_colnames=True)
from mlxtend.frequent_patterns import association_rules
#Tạo luật từ tập mục phổ biến với độ đo lift tối thiểu là 1
res=association_rules(res, metric="lift", min_threshold=1)
#In kết quả
res
