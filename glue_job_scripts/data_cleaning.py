import sys
from awsglue.context import GlueContext
from pyspark.context import SparkContext
from pyspark.sql.functions import col, window, avg, round, date_trunc, lit
from awsglue.utils import getResolvedOptions

args = getResolvedOptions(sys.argv, ["output_bucket"])
output_bucket = args["output_bucket"]
output_base = f"s3://{output_bucket}/cleaned/"

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

# Read all CSVs from the extracted S3 folder
df = spark.read.option("header", "true").csv(f"s3://{output_bucket}/iot_extracted_raw/")
df = df.withColumn("timestamp", col("timestamp").cast("timestamp"))

print(df)