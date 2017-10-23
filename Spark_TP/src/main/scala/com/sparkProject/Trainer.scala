package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover, CountVectorizer, IDF}


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

   /** CHARGER LE DATASET **/
   // 1
   val df_cleaned = spark
     .read
     .load("data/train_cleaned/*.parquet")
    println(df_cleaned.count)
    df_cleaned.select("text").show(5)


    /** TF-IDF **/
    // 1
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")
    val df_tokenized = tokenizer.transform(df_cleaned)


    // 2
    val stopworder = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("without_stopwords")
    val df_without_stopwords = stopworder.transform(df_tokenized)


    // 3
    val vectorizer = new CountVectorizer()
      .setInputCol("without_stopwords")
      .setOutputCol("vectorize")
    val vectorizeModel = vectorizer.fit(df_without_stopwords)
    val df_vectorized = vectorizeModel.transform(df_without_stopwords)
    df_vectorized.show(5)


    // 4
    val idf = new IDF()
      .setInputCol("vectorize")
      .setOutputCol("tfidf")
    val idfModel = idf.fit(df_vectorized)
    val df_tfidf = idfModel.transform(df_vectorized)

    df_tfidf.select("text", "without_stopwords", "tfidf").show(5)


    /** VECTOR ASSEMBLER **/


    /** MODEL **/


    /** PIPELINE **/


    /** TRAINING AND GRID-SEARCH **/

  }
}
