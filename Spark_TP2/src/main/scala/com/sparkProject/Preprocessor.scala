package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.SparkSession


object Preprocessor {

  def main(args: Array[String]): Unit = {

    val mySpark = SparkSession
      .builder()
      .appName("Spark SQL basic example")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()

    // For implicit conversions like converting RDDs to DataFrames
    import mySpark.implicits._

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 2-3
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/


    /** CHARGEMENT DES DONNÉES **/
    // 1
    val df_recovered = spark
      .read
      .option("header", true)
      .option("nullValue", false)
      .csv("data/train.csv")

    /*val df = spark.read.text("data/train.csv")
    val df2 = df.withColumn("replaced", regexp_replace($"value", "\"{2,}", " "))
    df2.write.format("csv").save("data/lolilol.csv")
    df2.select("replaced").show()*/

    // 2
    println(df_recovered.count(), df_recovered.columns.length)

    // 3
    df_recovered.show()

    // 4
    df_recovered.printSchema()

    // 5
    val df_casted = df_recovered
      .withColumn("goal", 'goal.cast("Int"))
      .withColumn("backers_count", 'backers_count.cast("Int"))
      .withColumn("final_status", 'final_status.cast("Int"))
    df_casted.printSchema()


    /** CLEANING **/
    // 1
    df_casted.describe("goal", "backers_count", "final_status").show()

    // 2
    import org.apache.spark.sql.functions._
    df_casted.groupBy("disable_communication").count().sort(desc("count")).show()

    // 3
    val df_drop_column = df_casted.drop("disable_communication")


    // 4
    val df_drop_futur = df_drop_column
      .drop("backers_count")
      .drop("state_changed_at")

    // 5


  }

}
