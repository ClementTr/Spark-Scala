package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}



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
    // On pourrait penser que "currency" et "country" sont redondantes, auquel cas on pourrait enlever une des colonne.
    // Mais en y regardant de plus près:
    //   - dans la zone euro: même monnaie pour différents pays => garder les deux colonnes.
    //   - il semble y avoir des inversions entre ces deux colonnes et du nettoyage à faire en utilisant les deux colonnes.
    //     En particulier on peut remarquer que quand country=false le country à l'air d'être dans currency:

    df_drop_futur.filter($"country".isNull).groupBy("currency").count.orderBy($"count".desc).show(50)

    def udf_country = udf{(country: String, currency: String) =>
      if (country == null) // && currency != "false")
        currency
      else
        country //: ((String, String) => String)  pour éventuellement spécifier le type
    }

    def udf_currency = udf{(currency: String) =>
      if ( currency != null && currency.length != 3 )
        null
      else
        currency //: ((String, String) => String)  pour éventuellement spécifier le type
    }


    val dfCountry: DataFrame = df_drop_futur
      .withColumn("country2", udf_country($"country", $"currency"))
      .withColumn("currency2", udf_currency($"currency"))
      .drop("country", "currency")

    dfCountry.groupBy("country2", "currency2").count.orderBy($"count".desc).show(50)


    // Pour aider notre algorithme, on souhaite qu'un même mot écrit en minuscules ou majuscules ne soit pas deux
    // "entités" différentes. On met tout en minuscules
    val dfLower: DataFrame = dfCountry
      .withColumn("name", lower($"name"))
      .withColumn("desc", lower($"desc"))
      .withColumn("keywords", lower($"keywords"))

    dfLower.show(50)


    /** FEATURE ENGINEERING: Ajouter et manipuler des colonnes **/
    // 1
    val df_day_campaign: DataFrame = dfLower
      .withColumn("deadline2", from_unixtime($"deadline"))
      .withColumn("launched_at2", from_unixtime($"launched_at"))
      .withColumn("days_campaign", datediff($"deadline2", $"launched_at2"))

    df_day_campaign.select("days_campaign").show(5)

    // 2
    val df_hours_prepa : DataFrame = df_day_campaign
      .withColumn("created_at2", from_unixtime($"created_at"))
      .withColumn("hours_prepa", round(($"launched_at" - $"created_at")/3600.0, 3)) // here timestamps are in seconds, there are 3600 seconds in one hour
      .filter($"hours_prepa" >= 0 && $"days_campaign" >= 0)

    df_hours_prepa.groupBy($"hours_prepa").count.orderBy($"count".desc).show()

    // 3
    val dfDurations : DataFrame = df_hours_prepa
      .drop("created_at", "deadline", "launched_at")

    // 4
    val dfText= dfDurations
      .withColumn("text", concat_ws(" ", $"name", $"desc", $"keywords"))

    /** VALEURS NULLES **/

    val dfReady: DataFrame = dfText
      .filter($"goal" > 0)
      .na
      .fill(Map(
        "days_campaign" -> -1,
        "hours_prepa" -> -1,
        "goal" -> -1
      ))

    // vérifier l'équilibrage pour la classification
    dfReady.groupBy("final_status").count.orderBy($"count".desc).show()


    // filtrer les classes qui nous intéressent
    // Final status contient d'autres états que Failed ou Succeed. On ne sait pas ce que sont ces états,
    // on peut les enlever ou les considérer comme Failed également. Seul "null" est ambigue et on les enlève.
    val dfFiltered = dfReady.filter($"final_status".isin(0, 1))
    println(dfFiltered.count)


    /** WRITING DATAFRAME **/

    dfFiltered.write.mode(SaveMode.Overwrite).parquet("data/train_cleaned")

  }

}
