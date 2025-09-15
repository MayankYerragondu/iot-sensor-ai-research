import sys
import json
from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql.functions import col, from_json, to_json, udf, expr, when, lit
from pyspark.sql.types import *
from pyspark.sql import SparkSession


args = getResolvedOptions(sys.argv, ["output_bucket"])
output_bucket = args["output_bucket"]

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

# collections = ["events_202503"]
collections = ["test_3"]

logger = glueContext.get_logger()


# custom schema for dynamic frame
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

# Define filter using MongoDB aggregation pipeline
filter_pipeline = [
  { "$match": {
      "payload.objects.0.data.0.devid": { "$exists": True },
      "payload.objects.0.data.0.states": { "$exists": True },
      "$or": [
        { "payload.objects.0.data.0.states.PirAlarm.pirAlarm": True },
        { "payload.objects.0.data.0.states.Humidity": { "$exists": True } },
        { "payload.objects.0.data.0.states.Luminance": { "$exists": True } },
        { "payload.objects.0.data.0.states.ContactAlarm.csAlarm": True },
        { "payload.objects.0.data.0.states.Temperature": { "$exists": True } }
      ]
  }}
]

for coll in collections:
    logger.info("*******loading {coll} collection data **********")
    # df = glueContext.create_dynamic_frame.from_options(
    #     connection_type="mongodb",
    #     connection_options={
    #         "connectionName": "xx-mongodb-connection",
    #         "database": "xx", 
    #         "collection": coll,
    #         "aggregation.pipeline": json.dumps(filter_query),
    #     }
    # ).toDF()
    mongo_uri = f"mongodb://.{coll}"

    spark = SparkSession.builder \
        .appName("MongoRead") \
        .getOrCreate()

    df = spark.read.format("mongodb") \
        .option("connection.uri", mongo_uri) \
        .option("database", "xx") \
        .option("collection", coll) \
        .option("aggregation.pipeline", json.dumps(filter_pipeline)) \
        .schema(custom_schema) \
        .load()

    # parse json string to object 
    # df = df.select(
    #     col("ts"),
    #     col("payload")
    # )
    # logger.info(f"finish loading collection {coll} data ")
    logger.info(f"================================")
    logger.info(f"df schema data : {df._jdf.schema().treeString()}")
    logger.info(f"================================")


    flat = df.select(
        col("ts").cast("timestamp").alias("timestamp"),
        col("payload.objects")[0]["data"][0]["devid"].alias("devid"),
        col("payload.objects")[0]["data"][0]["states"].alias("sensor_data")
    )

    def extract_nested_value_as_float(sensor_map, sensor_type, value_key):
        if sensor_map is None:
            return None
        try:
            sensor = sensor_map[sensor_type]
            value = sensor[value_key]
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                if isinstance(value, bool):
                    return 1.0 if value else 0.0
                return None
        except Exception:
            return None

    extract_value_udf = udf(extract_nested_value_as_float, FloatType())

    flat.printSchema()
    result = flat.select(
        "timestamp", "devid",
        # PirAlarm - boolean
        when(
            col("sensor_data").getItem("PirAlarm").isNotNull(),
            col("sensor_data").getItem("PirAlarm").getItem("pirAlarm")
        ).alias("pirAlarm"),
        
        # ContactAlarm - boolean
        when(
            col("sensor_data").getItem("ContactAlarm").isNotNull(),
            col("sensor_data").getItem("ContactAlarm").getItem("csAlarm")
        ).alias("csAlarm"),
        
        # Temperature - handle float/int conversion
        extract_value_udf(
            col("sensor_data"), 
            lit("Temperature"), 
            lit("temperature")
        ).alias("temperature"),
        
        # Humidity - handle float/int conversion
        extract_value_udf(
            col("sensor_data"), 
            lit("Humidity"), 
            lit("humidity")
        ).alias("humidity"),
        
        # Luminance - handle float/int conversion
        extract_value_udf(
            col("sensor_data"), 
            lit("Luminance"), 
            lit("lux")
        ).alias("lux")
    )

    logger.info(f"result schema data : {result._jdf.schema().treeString()}")

    result.show(n=10, truncate=False)
    

    # save to extracted data to S3 
    result.write.mode("append").option("header", "true").csv(f"s3://{output_bucket}/iot_extracted_raw/")
