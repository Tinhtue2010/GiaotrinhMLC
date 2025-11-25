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
from pyspark.ml.evaluation import ClusteringEvaluator

#Thử nghiệm phân cụm với số lượng cụm là 3
KMeans_=KMeans(featuresCol='iris_features', k=3) 
KMeans_Model=KMeans_.fit(assembled_data)
KMeans_Assignments=KMeans_Model.transform(assembled_data)

from pyspark.ml.feature import PCA as PCAml
pca = PCAml(k=2, inputCol="iris_features", outputCol="pca")
pca_model = pca.fit(assembled_data)
pca_transformed = pca_model.transform(assembled_data)

import numpy as np
x_pca = np.array(pca_transformed.rdd.map(lambda row: row.pca).collect())

cluster_assignment = np.array(KMeans_Assignments.rdd.map(lambda row: row.prediction).collect()).reshape(-1,1)
import seaborn as sns
import matplotlib.pyplot as plt
pca_data = np.hstack((x_pca,cluster_assignment))
#Vẽ đồ thị phân bố của các phần tử dữ liệu ở các cụm
pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal","cluster_assignment"))
sns.FacetGrid(pca_df,hue="cluster_assignment", height=6).map(plt.scatter, '1st_principal', '2nd_principal' ).add_legend()
plt.show()

#Tìm số lượng cụm tối ưu bằng cách thử nghiệm phân cụm với số lượng cụm từ 2 đến 11 để xem giá trị của 
# Silhouette thay đổi như thế nào và số cụm là bao nhiêu thì tốt
silhouette_scores=[]
evaluator = ClusteringEvaluator(featuresCol='iris_features', \
metricName='silhouette', distanceMeasure='squaredEuclidean')

for K in range(2,11):
    KMeans_=KMeans(featuresCol='iris_features', k=K)
    KMeans_fit=KMeans_.fit(assembled_data)
    KMeans_transform=KMeans_fit.transform(assembled_data) 
    evaluation_score=evaluator.evaluate(KMeans_transform)
    silhouette_scores.append(evaluation_score)
    
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1, figsize =(10,8))
ax.plot(range(2,11),silhouette_scores)
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Silhouette Score')
