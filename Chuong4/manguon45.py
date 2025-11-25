from pyspark.sql.functions import * 
from pyspark.ml import Pipeline 
from pyspark.ml.feature import VectorAssembler 
from pyspark.ml.feature import StringIndexer 
from pyspark.ml.classification import NaiveBayes 
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
#Đọc dữ liệu từ file vehicle_stolen_dataset.csv 
vehicle_data = pd.read_csv("./vehicle_stolen_dataset.csv", header=None)

#Khởi tạo Spark 
spark = SparkSession.builder.master("local[*]").getOrCreate() 

#Tạo DataFrame trên Spark
vehicle_df = spark.createDataFrame(vehicle_data) 
vehicle_df.show(2)

#Đổi tên các cột
vehicle_df = vehicle_df.select(col("0").alias("number_plate"),  col("1").alias("brand"),  
col("2").alias("color"),  
col("3").alias("time"),  
col("4").alias("stoled"))

#Chuyển dữ liệu từ chuỗi sang số
indexers = [
StringIndexer(inputCol="brand", outputCol = "brand_index"),  
StringIndexer(inputCol="color", outputCol = "color_index"),  
StringIndexer(inputCol="time", outputCol = "time_index"),  
StringIndexer(inputCol="stoled", outputCol = "label")]
pipeline = Pipeline(stages=indexers) 
indexed_vehicle_df = pipeline.fit(vehicle_df).transform(vehicle_df) 
indexed_vehicle_df.show(2,False) 

#Trường biển số không được sử dụng nên bỏ ra khỏi tập đặc trưng
vectorAssembler = VectorAssembler(
    inputCols = ["brand_index", "color_index", "time_index"], outputCol = "features") 
vindexed_vehicle_df = vectorAssembler.transform(indexed_vehicle_df) 
vindexed_vehicle_df.show(2, False)

#Bỏ các cột không còn dùng
vindexed_vehicle_df =vindexed_vehicle_df.drop("brand_index")
vindexed_vehicle_df =vindexed_vehicle_df.drop("color_index")
vindexed_vehicle_df =vindexed_vehicle_df.drop("time_index")
vindexed_vehicle_df =vindexed_vehicle_df.drop("brand")
vindexed_vehicle_df =vindexed_vehicle_df.drop("color")
vindexed_vehicle_df =vindexed_vehicle_df.drop("time")
vindexed_vehicle_df =vindexed_vehicle_df.drop("number_plate")
vindexed_vehicle_df =vindexed_vehicle_df.drop("stoled")
indexed_vehicle_df.show(2,False)

#Chia tập dữ liệu, số 42 là giá trị để sinh số ngẫu nhiên
splits = vindexed_vehicle_df.randomSplit([0.6,0.4], 42)  
train_df = splits[0] 
test_df = splits[1]

#Khởi tạo bộ phân lớp Naïve Bayes 
nb = NaiveBayes(modelType="multinomial")

#Huấn luyện mô hình
nbmodel = nb.fit(train_df)
#Dự đoán trên tập dữ liệu kiểm thử
predictions_df = nbmodel.transform(test_df)
predictions_df.show(2, True)

#Đánh giá hiệu năng phân lớp
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy") 
nbaccuracy = evaluator.evaluate(predictions_df) 
print("Test accuracy = " + str(nbaccuracy))
