from pyspark.sql import SparkSession 
from pyspark.sql.functions import * 
from pyspark.ml.feature import VectorAssembler 
from pyspark.ml.classification import NaiveBayes 
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd
from sklearn.datasets import load_iris

spark = SparkSession.builder \
    .appName("arlUsingPyspark") \
    .getOrCreate()

iris = load_iris(as_frame=True)
pddf_iris  = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
iris_spdf = spark.createDataFrame(pddf_iris) 

#In thông tin cấu trúc tập dữ liệu
iris_spdf.printSchema()
iris_spdf.show(2)

#Đổi tên các cột 
iris_spdf = iris_spdf.select(
col("sepal length (cm)").alias("sepallength"),  
col("sepal width (cm)").alias("sepalwidth"),  
col("petal length (cm)").alias("petallength"),  
col("petal width (cm)").alias("petalwidth"),  
col("target").alias("label"))

iris_spdf.printSchema()
iris_spdf.show(2)

#Chuyển dạng dữ liệu
from pyspark.ml.feature import VectorAssembler
#We have given False for turn off default truncation
vectorAssembler = VectorAssembler(
    inputCols = ["sepallength", "sepalwidth", "petallength","petalwidth"], outputCol = "features") 
#Xóa các cột không còn dùng nữa
feature_iris_spdf = vectorAssembler.transform(iris_spdf) 
feature_iris_spdf=feature_iris_spdf.drop("sepallength")
feature_iris_spdf=feature_iris_spdf.drop("sepalwidth")
feature_iris_spdf=feature_iris_spdf.drop("petallength")
feature_iris_spdf=feature_iris_spdf.drop("petalwidth")
feature_iris_spdf.show(5, False)
splits = feature_iris_spdf.randomSplit([0.67,0.33], 42) 
# optional value 42 is seed for sampling 
train_df = splits[0] 
test_df = splits[1]
#Khởi tạo bộ phân lớp Naïve Bayes 
nb = NaiveBayes(modelType="multinomial")
#Huấn luyện mô hình
nbmodel = nb.fit(train_df)
#Phân lớp dữ liệu kiểm thử
predictions_df = nbmodel.transform(test_df)
predictions_df.show(5, True)

#Đánh giá hiệu năng phân lớp
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy") 
nbaccuracy = evaluator.evaluate(predictions_df) 
print("Test accuracy = " + str(nbaccuracy))
