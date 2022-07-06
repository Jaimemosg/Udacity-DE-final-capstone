import configparser
import logging
import os
from typing import List

import pandas as pd
from boto3.session import Session
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, lit
from pyspark.sql.functions import min as ps_min
from pyspark.sql.functions import (
    monotonically_increasing_id,
    month,
    split,
    to_date,
    udf,
    year,
)
from pyspark.sql.types import DateType, FloatType

logger = logging.getLogger()
logger.setLevel(logging.INFO)

config = configparser.ConfigParser()
config.read("config.cfg", encoding="utf-8-sig")

os.environ["AWS_ACCESS_KEY_ID"] = config["AWS"]["AWS_ACCESS_KEY_ID"]
os.environ["AWS_SECRET_ACCESS_KEY"] = config["AWS"]["AWS_SECRET_ACCESS_KEY"]

SOURCE_S3_BUCKET = config["S3"]["SOURCE_S3_BUCKET"]
TARGET_S3_BUCKET = config["S3"]["TARGET_S3_BUCKET"]
ACCESS_KEY = config["AWS"]["AWS_ACCESS_KEY_ID"]
SECRET_KEY = config["AWS"]["AWS_SECRET_ACCESS_KEY"]

session = Session(
    aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY
)
s3 = session.resource("s3")


def create_spark_session() -> SparkSession:
    """Creates a spark session.

    Returns:
        SparkSession: Spark session needed for run code.
    """
    # spark = (
    #     SparkSession.builder.config(
    #         "spark.jars.packages", "saurfang:spark-sas7bdat:2.0.0-s_2.11"
    #     )
    #     .enableHiveSupport()
    #     .getOrCreate()
    # )
    spark = (
        SparkSession.builder.config(
            "spark.jars.repositories", "https://repos.spark-packages.org/"
        )
        .config(
            "spark.hadoop.fs.s3a.impl",
            "org.apache.hadoop.fs.s3a.S3AFileSystem",
        )
        .config(
            "spark.jars.packages",
            "saurfang:spark-sas7bdat:2.0.0-s_2.11,"
            "org.apache.hadoop:hadoop-aws:3.2.2,"
            "io.delta:delta-core_2.12:1.1.0,"
            "com.amazonaws:aws-java-sdk-bundle:1.12.180",
        )
        .enableHiveSupport()
        .getOrCreate()
    )
    # spark = SparkSession.builder.config(
    #     "spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0"
    # ).getOrCreate()
    spark._jsc.hadoopConfiguration().set(
        "com.amazonaws.services.s3.enableV4", "true"
    )
    spark._jsc.hadoopConfiguration().set(
        "fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem"
    )
    spark._jsc.hadoopConfiguration().set(
        "fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.InstanceProfileCredentialsProvider,com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
    )
    spark._jsc.hadoopConfiguration().set(
        "fs.AbstractFileSystem.s3a.impl", "org.apache.hadoop.fs.s3a.S3A"
    )

    return spark


def sas_to_datetime(date: pd.Series) -> pd.Series:
    """Transforms arrdate, depdate from SAS time format to pandad.datetime

    Args:
        date (pd.Series): Series to be converted to pandas.datetime.

    Returns:
        pd.Series: Series converted.
    """
    if date is not None:
        return pd.to_timedelta(date, unit="D") + pd.Timestamp("1960-1-1")


sas_to_date_udf = udf(sas_to_datetime, DateType())


def rename_columns(table: DataFrame, new_columns: List[str]) -> DataFrame:
    """Change columns names from a Spark DataFrame

    Args:
        - table (DataFrame): original table.
        - new_columns (List[str]): new columns names. It must be ordered (i.e,
        must keep the same order of the table columns names).

    Returns:
        DataFrame: table with names changed.
    """
    for original, new in zip(table.columns, new_columns):
        table = table.withColumnRenamed(original, new)
    return table


def process_immigration_data(
    spark: SparkSession, input_path: str, output_path: str
) -> None:
    """Process immigration data to get fact_immigration, dim_immi_personal and
    dim_immi_airline tables.

        Arguments:
            - spark (SparkSession): SparkSession object.
            - input_path (str): Source S3 endpoint.
            - output_path (str): Target S3 endpoint.
    """

    logging.info("Start processing immigration (fact table)")

    immigration_path = input_path + "18-83510-I94-Data-2016/*.sas7bdat"

    df = spark.read.format("com.github.saurfang.sas.spark").load(
        immigration_path
    )

    logging.info("Start processing fact_immigration")

    fact_immigration = (
        df.select(
            "cicid",
            "i94yr",
            "i94mon",
            "i94port",
            "i94addr",
            "arrdate",
            "depdate",
            "i94mode",
            "i94visa",
        )
        .distinct()
        .withColumn("immigration_id", monotonically_increasing_id())
    )

    new_columns = [
        "cic_id",
        "year",
        "month",
        "city_code",
        "state_code",
        "arrive_date",
        "departure_date",
        "mode",
        "visa",
    ]
    fact_immigration = rename_columns(fact_immigration, new_columns)

    fact_immigration = fact_immigration.withColumn(
        "country", lit("United States")
    )
    fact_immigration = fact_immigration.withColumn(
        "arrive_date", sas_to_date_udf(col("arrive_date"))
    )
    fact_immigration = fact_immigration.withColumn(
        "departure_date", sas_to_date_udf(col("departure_date"))
    )

    fact_immigration.write.mode("overwrite").partitionBy("state_code").parquet(
        path=f"{output_path}/fact_immigration"
    )

    # Get min date
    min_arrive = fact_immigration.select(ps_min("arrive_date")).collect()
    min_departure = fact_immigration.select(ps_min("departure_date")).collect()
    min_date = ps_min(min_arrive, min_departure)

    logging.info("Start processing immigrant table")

    immigrant = (
        df.select("cicid", "i94cit", "i94res", "biryear", "gender", "insnum")
        .distinct()
        .withColumn("immi_personal_id", monotonically_increasing_id())
    )

    new_columns = [
        "cic_id",
        "citizen_country",
        "residence_country",
        "birth_year",
        "gender",
        "ins_num",
    ]
    immigrant = rename_columns(immigrant, new_columns)

    immigrant.write.mode("overwrite").parquet(path=f"{output_path}/immigrant")

    logging.info("Start processing immigration_airline table")

    immigration_airline = (
        df.select("cicid", "airline", "admnum", "fltno", "visatype")
        .distinct()
        .withColumn("immi_airline_id", monotonically_increasing_id())
    )

    new_columns = [
        "cic_id",
        "airline",
        "admin_num",
        "flight_number",
        "visa_type",
    ]
    immigration_airline = rename_columns(immigration_airline, new_columns)

    immigration_airline.write.mode("overwrite").parquet(
        path=f"{output_path}/dim_immi_airline"
    )

    return min_date


def process_airport_table(
    spark: SparkSession, input_path: str, output_path: str
) -> None:
    # Upload airport table
    airport_path = f"{input_path}/airport-codes_csv.csv"

    df_airport_codes = spark.read.csv(airport_path)
    split_col = split(df_airport_codes["coordinates"], ",")

    airport_codes = (
        df_airport_codes.withColumn(
            "lat", split_col.getItem(0).cast(FloatType())
        )
        .withColumn("long", split_col.getItem(1).cast(FloatType()))
        .drop(df_airport_codes.coordinates)
    )

    print(airport_codes.show(3, vertical=True))

    airport_codes.write.mode("overwrite").parquet(
        path=f"{output_path}/airport_codes"
    )


def process_temp_table(
    spark: SparkSession, input_path: str, output_path: str, min_date
) -> None:
    # Upload temperature table
    temp_path = f"{input_path}/GlobalLandTemperaturesByCity.csv"
    df_temp = spark.read.csv(temp_path)
    temperature = (
        df_temp.select(
            to_date(df_temp.dt).alias("dt"),
            "AverageTemperature",
            "AverageTemperatureUncertainty",
            "City",
            "Country",
        )
        .filer(df_temp.Country == "United States")
        .witthColumn("year", year("dt"))
        .witthColumn("month", month("dt"))
        .withColumnRenamed("AverageTemperature", "avg_temp")
        .withColumnRenamed(
            "AverageTemperatureUncertainty", "avg_temp_uncertnty"
        )
        .withColumnRenamed("City", "city")
        .withColumnRenamed("Country", "country")
    ).filter(df_temp.dt >= min_date)

    temperature.write.mode("overwrite").partitionBy("year").parquet(
        path=f"{output_path}/temperature"
    )
    print(temperature.show(3, vertical=True))


def process_label_descriptions(
    spark: SparkSession, input_path: str, output_path: str
) -> None:
    """Parsing label desctiption file to extract codes of country, city, state

    Arguments:
        - spark (SparkSession): SparkSession object.
        - input_path (str): Source S3 endpoint.
        - output_path (str): Target S3 endpoint.
    """

    logging.info("Start processing label descriptions")
    label_file = f"{input_path}/I94_SAS_Labels_Descriptions.SAS"
    with open(label_file) as f:
        contents = f.readlines()

    country_code = {}
    for countries in contents[10:298]:
        pair = countries.split("=")
        code, country = pair[0].strip(), pair[1].strip().strip("'")
        country_code[code] = country
    spark.createDataFrame(
        country_code.items(), ["code", "country"]
    ).write.mode("overwrite").parquet(path=output_path + "country_code")

    city_code = {}
    for cities in contents[303:962]:
        pair = cities.split("=")
        code, city = (
            pair[0].strip("\t").strip().strip("'"),
            pair[1].strip("\t").strip().strip("''"),
        )
        city_code[code] = city
    spark.createDataFrame(city_code.items(), ["code", "city"]).write.mode(
        "overwrite"
    ).parquet(path=f"{output_path}/city_code")

    state_code = {}
    for states in contents[982:1036]:
        pair = states.split("=")
        code, state = (
            pair[0].strip("\t").strip("'"),
            pair[1].strip().strip("'"),
        )
        state_code[code] = state
    spark.createDataFrame(state_code.items(), ["code", "state"]).write.mode(
        "overwrite"
    ).parquet(path=f"{output_path}/state_code")


def main():
    """Runs the whole process
    - Creates Spark session.
    - Get fact and dimension tables (a star schema is regarded).
    - Extract some features (country, city, state).
    """
    logging.info("Data processing starts!")
    spark = create_spark_session()
    input_path = SOURCE_S3_BUCKET
    output_path = TARGET_S3_BUCKET

    min_date = process_immigration_data(spark, input_path, output_path)
    process_temp_table(spark, input_path, output_path, min_date)
    process_label_descriptions(spark, input_path, output_path)
    logging.info("Data processing completed!")


if __name__ == "__main__":
    main()
