from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, col, avg, length

spark = SparkSession.builder.appName("Gutenberg Metadata").getOrCreate()

books_df = spark.read.text("/root/gutenberg/*.txt") \
    .withColumnRenamed("value", "text") \
    .withColumn("file_name", col("input_file_name()"))

metadata_df = books_df \
    .withColumn("title", regexp_extract("text", r"Title:\s*(.*)", 1)) \
    .withColumn("author", regexp_extract("text", r"Author:\s*(.*)", 1)) \
    .withColumn("release_date", regexp_extract("text", r"Release Date:\s*(.*)", 1)) \
    .withColumn("language", regexp_extract("text", r"Language:\s*(.*)", 1))

metadata_filtered = metadata_df.filter(
    (col("title") != "") | (col("author") != "")
)

metadata_filtered.select(
    avg(length("title")).alias("avg_title_length")
).show()

metadata_filtered.groupBy("language").count().show()
