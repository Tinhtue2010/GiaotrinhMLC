from pyspark.sql import SparkSession 
from pyspark.sql.functions import * 
from pyspark.ml import Pipeline 
from pyspark.ml.feature import VectorAssembler 
from pyspark.ml.feature import StringIndexer 
from pyspark.ml.classification import NaiveBayes 
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd

spark = SparkSession.builder \
    .appName("arlUsingPyspark") \
    .getOrCreate()

#Tải dữ liệu từ trang web về, trong đó bỏ đi 22 dòng đầu tiên không phải dữ liệu
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

#Vì dòng thông tin dài quá nên bị cắt thành 2 dòng, tiến hành ghép và tách phần dữ liệu và nhãn ra
x = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]

#Gộp lại cả dữ liệu và nhãn lại thành chung 1 frame, bổ sung các tên cột
pd_df  = pd.DataFrame(data= np.c_[x, y],
                     columns= ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PT', 'B', 'LSTAT','MV'])
#Chuyển sang Spark frame
spark_df = spark.createDataFrame(pd_df)
spark_df.printSchema()
spark_df.show(2)

#Chuyển dạng dữ liệu, gộp các đặc trưng vào một cột
from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(
    inputCols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PT', 'B', 'LSTAT'], 
    outputCol = 'features')
#Chỉ lấy 2 cột là features và MV bỏ qua các cột thông tin khác (thay vì xóa)
vhouse_df = vectorAssembler.transform(spark_df)
vhouse_df = vhouse_df.select(['features', 'MV'])
vhouse_df.show(3)

#Chia tập dữ liệu thành tập huấn luyện và kiểm thử theo tỷ lệ 70:30
splits = vhouse_df.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]

#Khởi tạo và huấn luyện
from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol = 'features', labelCol='MV', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)

#In ra các tham số sau khi huấn luyện
print("Coefficients: " + str(lr_model.coefficients))
#print("Intercept: " + str(lr_model.intercept))

#In thông tin các độ đo đánh giá
trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)
train_df.describe().show()

lr_predictions = lr_model.transform(test_df)
lr_predictions.select("prediction","MV","features").show(5)
from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="MV",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))
test_result = lr_model.evaluate(test_df)
print("Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)
print("numIterations: %d" % trainingSummary.totalIterations)
