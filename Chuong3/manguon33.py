#Import các thư viện
from pyspark import SparkContext
from pyspark.sql import functions as f, SparkSession, Column
from pyspark_dist_explore import hist
import matplotlib.pyplot as plt
from pyspark.ml.fpm import FPGrowth
#Tạo một phiên làm việc của Spark
spark = SparkSession.builder \
    .appName("arlUsingPyspark") \
    .getOrCreate()
#Tải dữ liệu và tạo thêm một cột id với giá trị tự tăng
df = spark.read.csv("Market_Basket_Optimisation.csv", header=False).withColumn("id", f.monotonically_increasing_id())
#Hiển thị một số dòng dữ liệu
df.show(2)
#In thông tin cấu trúc tập dữ liệu
df.printSchema()
df_basket = df.select("id", f.array([df[c] for c in df.columns[:11]]).alias("basket"))
df_basket.printSchema()
# Tham số False báo lệnh không bỏ qua tên các cột
df_basket.show(3, False) 
df_aggregated = df_basket.select("id", f.array_except("basket", f.array(f.lit(None))).alias("basket"))
df_aggregated.show(3, False)
# Run FPGrowth and fit the model.
fp = FPGrowth(minSupport=0.001, minConfidence=0.001, itemsCol='basket', predictionCol='prediction')
model = fp.fit(df_aggregated)
# View a subset of the frequent itemset. 
model.freqItemsets.show(10, False)
# Use filter to view just the association rules with the highest confidence.
model.associationRules.filter(model.associationRules.confidence>0.15).show(20, False)
