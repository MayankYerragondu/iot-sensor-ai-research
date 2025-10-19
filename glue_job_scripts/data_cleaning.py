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


# ========= PIR ALARM =========
pir_df = df.select("timestamp", "devid", "pirAlarm") \
            .where(col("pirAlarm").isNotNull()) \

pir_device_ids = pir_df.select("devid").distinct().rdd.map(lambda r: r[0]).collect()

for devid in pir_device_ids:
    single_device_df = pir_df.filter(col("devid") == devid) \
        .select(
            col("devid"),
            col("timestamp")
        )
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
        )
    output_path = f"{output_base}contact_alarm/{devid}/"
    single_device_df.coalesce(1) \
        .write.mode("append") \
        .option("header", "true") \
        .csv(output_path)


# ========= Env Sensor =========
# Environment sensor: aggregate over 5-minute windows
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


# Write one folder per device
device_ids = agg_df.select("devid").distinct().rdd.map(lambda r: r[0]).collect()

for devid in device_ids:
    device_df = agg_df.filter(col("devid") == devid) \
        .select(
            "devid", 
            "window_start", 
            "window_end",
            "avg_temperature", 
            "avg_humidity", 
            "avg_lux"
        )

    output_path = f"{output_base}env_sensor/{devid}/"

    device_df.coalesce(1) \
        .write.mode("append") \
        .option("header", "true") \
        .csv(output_path)