import pandas as pd
from sklearn.datasets import load_iris
from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .appName("arlUsingPyspark") \
    .getOrCreate()

df_iris = load_iris(as_frame=True)
pd_df_iris = pd.DataFrame(df_iris.data, columns = df_iris.feature_names)
pd_df_iris['target'] = pd.Series(df_iris.target)
spark_df_iris = spark.createDataFrame(pd_df_iris)
spark_df_iris = spark_df_iris.drop("target")


#In thông tin cấu trúc tập dữ liệu
spark_df_iris.printSchema()
spark_df_iris.show(2)

#Chuyển dạng dữ liệu
from pyspark.ml.feature import VectorAssembler
assemble=VectorAssembler(inputCols=[
'sepal length (cm)',
'sepal width (cm)',
'petal length (cm)',
'petal width (cm)'],outputCol = 'iris_features')
assembled_data=assemble.transform(spark_df_iris)

from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
#Tạo phần tử dữ liệu mới để dự báo
data = [(Vectors.dense([ 5.2, 3.4, 1.4, 0.2]), 2.0)]
tdf = spark.createDataFrame(data, ["features", "target"])

#Thử nghiệm phân cụm với số lượng cụm là 3
KMeans_=KMeans(featuresCol='iris_features', k=3) 
KMeans_Model=KMeans_.fit(assembled_data)
#Hiển thị dữ liệu của phần tử mới
tdf.show()
#Dự báo phần tử mới
print(KMeans_Model.predict(tdf.head().features))
#Dự báo phần gần với phần tử mới
print(KMeans_Model.predict(assembled_data.head().iris_features))
