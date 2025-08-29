from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.functions import expr
import sys
import json
from pyspark.context import SparkContext
from pyspark.sql.functions import col, from_json, to_json,udf, expr, size, map_keys
from pyspark.sql.types import *
from pyspark.sql.functions import when
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, MapType
from pyspark.sql.functions import col, window, avg, round, date_trunc, lit


# Define filter using MongoDB aggregation pipeline
custom_schema = StructType([
    StructField("ts", IntegerType()),
    StructField("payload", StructType([
        StructField("cmd", StringType()),
        StructField("reqid", StringType()),
        StructField("objects", ArrayType(
            StructType([
                StructField("type", StringType()),
                StructField("data", ArrayType(
                    StructType([
                        StructField("devid", StringType()),
                        StructField("states", MapType(StringType(), MapType(StringType(), StringType()))),
                        StructField("status", IntegerType())
                    ])
                ))
            ])
        ))
    ]))
])

filter_pipeline = []

spark = None

try:
    mongo_uri = "mongo-connect-url"

    conf = SparkConf()
    conf.set("spark.driver.bindAddress", "127.0.0.1")
    conf.set("spark.driver.host", "127.0.0.1")

    spark = SparkSession.builder \
        .appName("MongoRead") \
        .master("local[*]") \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.2.0") \
        .config("spark.mongodb.read.connection.uri", mongo_uri) \
        .config(conf=conf) \
        .getOrCreate()

    df = spark.read.format("mongodb") \
        .option("aggregation.pipeline", json.dumps(filter_pipeline)) \
        .schema(custom_schema) \
        .load()
    df.show(n=10)
except Exception as e:
    print(e)
finally:
    spark.stop()