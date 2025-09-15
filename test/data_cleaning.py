import sys
from pyspark.context import SparkContext
from pyspark.sql.functions import col, window, avg, round, date_trunc, lit
from pyspark.sql import SparkSession


output_bucket = "iot-glue-bucket-multi-model"
output_base = f"s3a://{output_bucket}/cleaned/"


# Configure Hadoop to use S3A in code

spark = SparkSession.builder \
    .appName("S3 Test") \
    .config("spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.4.1,"
            "org.apache.hadoop:hadoop-common:3.4.1") \
    .getOrCreate()

sc = spark.sparkContext

# S3 configuration
hadoop_conf = sc._jsc.hadoopConfiguration()
hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
hadoop_conf.set("fs.s3.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
hadoop_conf.set("fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
hadoop_conf.set("fs.s3a.endpoint", "s3.amazonaws.com")

# Read all CSVs from the extracted S3 folder
df = spark.read.option("header", "true").csv(f"s3a://{output_bucket}/output/iot_extracted_raw/")
df = df.withColumn("timestamp", col("timestamp").cast("timestamp"))

# # 1. Save pirAlarm data
# df.select("timestamp", "devid", "pirAlarm") \
#   .where(col("pirAlarm").isNotNull()) \
#   .coalesce(1) \
#   .write.mode("overwrite") \
#   .option("header", "true") \
#   .csv(f"s3://{output_bucket}/cleaned/pir_alarm/")

# # 2. Save csAlarm (contact alarm) data
# df.select("timestamp", "devid", "csAlarm") \
#   .where(col("csAlarm").isNotNull()) \
#   .coalesce(1) \
#   .write.mode("overwrite") \
#   .option("header", "true") \
#   .csv(f"s3://{output_bucket}/cleaned/contact_alarm/")


# ========= PIR ALARM =========
pir_df = df.select("timestamp", "devid", "pirAlarm") \
            .where(col("pirAlarm").isNotNull()) \

pir_device_ids = pir_df.select("devid").distinct().rdd.map(lambda r: r[0]).collect()

for devid in pir_device_ids:
    single_device_df = pir_df.filter(col("devid") == devid).select("timestamp")
    output_path = f"{output_base}pir_alarm/{devid}/"
    single_device_df.coalesce(1).write.mode("append").option("header", "true").csv(output_path)

# ========= CONTACT ALARM =========
contact_df = df.select("timestamp", "devid", "csAlarm") \
                .where(col("csAlarm").isNotNull()) \

contact_device_ids = contact_df.select("devid").distinct().rdd.map(lambda r: r[0]).collect()

for devid in contact_device_ids:
    # single_device_df = contact_df.filter(col("devid") == devid).select("timestamp")
    single_device_df = contact_df.filter(col("devid") == devid) \
        .select(
            col("devid"),
            col("timestamp"),
            date_trunc("hour", col("timestamp")).alias("hour"),
            lit(True).alias("contact_alarm")
        )
    output_path = f"{output_base}contact_alarm/{devid}/"
    single_device_df.coalesce(1) \
        .write.mode("append") \
        .option("header", "true") \
        .csv(output_path)

# 3. Environment sensor: aggregate over 5-minute windows
# Cast sensor fields to correct types
env_df = df.select(
    "timestamp", "devid",
    col("pirAlarm"),
    col("csAlarm"),
    col("temperature").cast("float"),
    col("humidity").cast("float"),
    col("lux").cast("float")
)
# Filter out rows with pirAlarm == true or csAlarm == true
env_df_filtered = env_df.filter(
    ((col("pirAlarm").isNull()) | (col("pirAlarm") != "true")) &
    ((col("csAlarm").isNull()) | (col("csAlarm") != "true"))
)

# aggregate over 5-minute windows
agg_df = env_df_filtered.groupBy(
    window("timestamp", "5 minutes"),
    "devid"
).agg(
    round(avg("temperature"), 2).alias("avg_temperature"),
    round(avg("humidity"), 2).alias("avg_humidity"),
    round(avg("lux"), 2).alias("avg_lux")
).select(
    col("devid"),
    col("window.start").alias("window_start"),
    col("window.end").alias("window_end"),
    col("avg_temperature"),
    col("avg_humidity"),
    col("avg_lux")
)


# 4. Write one folder per device
device_ids = agg_df.select("devid").distinct().rdd.map(lambda r: r[0]).collect()

for devid in device_ids:
    device_df = agg_df.filter(col("devid") == devid) \
        .select(
            "window_start","window_end","avg_temperature", "avg_humidity", "avg_lux"
        )

    output_path = f"{output_base}env_sensor/{devid}/"

    device_df.coalesce(1) \
        .write.mode("append") \
        .option("header", "true") \
        .csv(output_path)