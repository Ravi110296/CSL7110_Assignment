from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType
import numpy as np

from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.ml.feature import CountVectorizer, IDF

# Create Spark Session
spark = SparkSession.builder \
    .appName("Gutenberg_Assignment") \
    .getOrCreate()

# Q10 - LOAD DATASET

DATA_PATH = "/mnt/c/Users/ASUS/Downloads/Big Data/Assignment/D184MB"

books_df = (
    spark.read.text(DATA_PATH)
    .withColumn("file_name", input_file_name())
    .withColumn("file_name", F.regexp_extract("file_name", r'([^/]+$)', 1))
)

print("=== Schema ===")
books_df.printSchema()

print("=== Preview ===")
books_df.show(5, truncate=False)

# Q10 - METADATA EXTRACTION

# Title
title_df = books_df.filter(
    F.col("value").startswith("Title:")
).withColumn(
    "title",
    F.regexp_extract("value", r"Title:\s*(.*)", 1)
).withColumn(
    "title",
    F.trim("title")
).filter(
    F.col("title") != ""
)

# Release Year
release_df = books_df.filter(
    F.col("value").startswith("Release Date:")
).withColumn(
    "release_year",
    F.regexp_extract("value", r"(\d{4})", 1)
).filter(
    F.col("release_year") != ""
).withColumn(
    "release_year",
    F.col("release_year").cast("int")
)

# Language
language_df = books_df.filter(
    F.col("value").startswith("Language:")
).withColumn(
    "language",
    F.regexp_extract("value", r"Language:\s*(.*)", 1)
).withColumn(
    "language",
    F.trim("language")
).filter(
    F.col("language") != ""
)

# Combine metadata
metadata_df = title_df.select("file_name", "title") \
    .join(release_df.select("file_name", "release_year"),
          "file_name", "left") \
    .join(language_df.select("file_name", "language"),
          "file_name", "left")

print("=== Metadata Preview ===")
metadata_df.show(5, truncate=False)

# Q10 ANALYSIS

print("=== Books Per Year ===")
metadata_df.groupBy("release_year") \
    .count() \
    .orderBy("release_year") \
    .show()

print("=== Most Common Languages ===")
metadata_df.groupBy("language") \
    .count() \
    .orderBy(F.desc("count")) \
    .show(5)

print("=== Average Title Length ===")
metadata_df.withColumn(
    "title_length", F.length("title")
).select(
    F.avg("title_length")
).show()

# Q11 - TEXT CLEANING

books_full_df = books_df.groupBy("file_name") \
    .agg(F.concat_ws(" ", F.collect_list("value")).alias("text"))

# Remove Gutenberg headers
books_full_df = books_full_df.withColumn(
    "text",
    F.regexp_replace("text", r"\*\*\* START OF.*?\*\*\*", "")
)

books_full_df = books_full_df.withColumn(
    "text",
    F.regexp_replace("text", r"\*\*\* END OF.*?\*\*\*", "")
)

# Lowercase
books_full_df = books_full_df.withColumn(
    "text", F.lower("text")
)

# Remove punctuation
books_full_df = books_full_df.withColumn(
    "text",
    F.regexp_replace("text", r"[^a-z\s]", "")
)

print("=== Cleaned Text Preview ===")
books_full_df.show(3, truncate=200)

# Tokenization

tokenizer = Tokenizer(inputCol="text", outputCol="words")
words_df = tokenizer.transform(books_full_df)

remover = StopWordsRemover(inputCol="words", outputCol="filtered")
filtered_df = remover.transform(words_df)

print("=== Tokenized Preview ===")
filtered_df.select("file_name", "filtered").show(3, truncate=False)

# TF-IDF

cv = CountVectorizer(
    inputCol="filtered",
    outputCol="rawFeatures",
    vocabSize=5000,
    minDF=2
)

cv_model = cv.fit(filtered_df)
featurized_df = cv_model.transform(filtered_df)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idf_model = idf.fit(featurized_df)

tfidf_df = idf_model.transform(featurized_df)

print("=== TF-IDF Vector Preview ===")
tfidf_df.select("file_name", "features").show(5, truncate=False)

# Cosine Similarity (Compare One Book)

target_book = tfidf_df.select("file_name").first()[0]

target_vector = tfidf_df.filter(
    F.col("file_name") == target_book
).select("features").first()[0]

def cosine_similarity(vector):
    v1 = vector.toArray()
    v2 = target_vector.toArray()
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return float(np.dot(v1, v2) /
                 (np.linalg.norm(v1) * np.linalg.norm(v2)))

cosine_udf = F.udf(cosine_similarity, DoubleType())

similarity_df = tfidf_df.withColumn(
    "similarity",
    cosine_udf("features")
)

print("=== Top 5 Similar Books ===")
similarity_df.filter(F.col("file_name") != target_book) \
    .orderBy(F.desc("similarity")) \
    .select("file_name", "similarity") \
    .show(5)

# Q12 - AUTHOR INFLUENCE NETWORK

author_df = books_df.filter(
    F.col("value").startswith("Author:")
).withColumn(
    "author",
    F.regexp_extract("value", r"Author:\s*(.*)", 1)
).withColumn(
    "author", F.trim("author")
).filter(
    F.col("author") != ""
)

author_year_df = author_df.select("file_name", "author") \
    .join(release_df.select("file_name", "release_year"),
          "file_name", "inner")

# One row per author
author_year_unique = author_year_df.groupBy("author") \
    .agg(F.min("release_year").alias("first_release"))

# Influence window
X = 5

a = author_year_unique.alias("a")
b = author_year_unique.alias("b")

influence_df = a.join(
    b,
    (F.col("a.author") != F.col("b.author")) &
    (F.col("a.first_release") < F.col("b.first_release")) &
    ((F.col("b.first_release") - F.col("a.first_release")) <= X)
).select(
    F.col("a.author").alias("author1"),
    F.col("b.author").alias("author2")
).distinct()

print("=== Influence Edges Preview ===")
influence_df.show(10, truncate=False)

print("=== Top Influential Authors (Out-Degree) ===")
influence_df.groupBy("author1") \
    .count() \
    .orderBy(F.desc("count")) \
    .show(5)

print("=== Most Influenced Authors (In-Degree) ===")
influence_df.groupBy("author2") \
    .count() \
    .orderBy(F.desc("count")) \
    .show(5)

spark.stop()