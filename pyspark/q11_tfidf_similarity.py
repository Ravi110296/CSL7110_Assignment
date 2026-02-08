from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, regexp_replace, split, explode, log, sqrt, sum as spark_sum, col

spark = SparkSession.builder.appName("TFIDF Similarity").getOrCreate()

books_df = spark.read.text("/root/gutenberg/*.txt") \
    .withColumnRenamed("value", "text") \
    .withColumn("file_name", col("input_file_name()"))

clean_df = books_df.withColumn(
    "clean_text",
    regexp_replace(lower("text"), "[^a-z\\s]", "")
)

words_df = clean_df.withColumn(
    "word", explode(split("clean_text", "\\s+"))
).filter("word != ''")

tf_df = words_df.groupBy("file_name", "word").count() \
    .withColumnRenamed("count", "tf")

num_docs = books_df.select("file_name").distinct().count()

df_df = tf_df.groupBy("word").count() \
    .withColumnRenamed("count", "df")

idf_df = df_df.withColumn("idf", log(num_docs / col("df")))

tfidf_df = tf_df.join(idf_df, "word") \
    .withColumn("tfidf", col("tf") * col("idf"))
