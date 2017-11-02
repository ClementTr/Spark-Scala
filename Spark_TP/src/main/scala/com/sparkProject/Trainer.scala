package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.feature.{CountVectorizer, IDF, RegexTokenizer, StopWordsRemover}
import org.apache.spark.ml.feature.VectorAssembler


object Trainer {

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
    val df_vectorized = vectorizer
      .fit(df_without_stopwords)
      .transform(df_without_stopwords)
    df_vectorized.show(5)


    // 4
    val idf = new IDF()
      .setInputCol("vectorize")
      .setOutputCol("tfidf")
    val df_tfidf = idf
      .fit(df_vectorized)
      .transform(df_vectorized)
    df_tfidf.select("text", "without_stopwords", "tfidf").show(5)


    // 5
    val indexer_country = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
    val df_country_indexed =
      indexer_country
        .fit(df_tfidf)
        .transform(df_tfidf)


    // 6
    val indexer_currency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
    val df_country_currency_indexed =
      indexer_currency
        .fit(df_country_indexed)
        .transform(df_country_indexed)


    // 7
    val assembler = new VectorAssembler()
      .setInputCols(Array(
        "tfidf", "days_campaign", "hours_prepa", "goal",
        "country_indexed", "currency_indexed"))
      .setOutputCol("features")
    val df_assembled = assembler.transform(df_country_currency_indexed)


    // 8
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)


    // 9 -- Pipeline
    val pipeline = new Pipeline()
      .setStages(Array(
        tokenizer,
        stopworder,
        vectorizer,
        idf,
        indexer_country,
        indexer_currency,
        assembler,
        lr
      ))


    //10
    val df_init = df_cleaned
      .withColumn("final_status", 'final_status.cast("Double"))
    val Array(training, test) = df_init
      .select("text", "goal", "country2", "currency2",
        "deadline2", "launched_at2", "days_campaign",
        "created_at2", "hours_prepa", "final_status")
      .randomSplit(Array(0.9, 0.1))

    //11
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(
        scala.math.pow(10,-8),
        scala.math.pow(10,-6),
        scala.math.pow(10,-4),
        scala.math.pow(10,-2)))
      .addGrid(vectorizer.minDF, Array(55.0, 75.0, 85.0, 95.0)).build()

    val eval = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(eval)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    val model = trainValidationSplit.fit(training)
    val df_WithPredictions = model.transform(test)

    //12
    val score = eval.evaluate(df_WithPredictions)
    print("My score: ", score)

    // 13
    df_WithPredictions.groupBy("final_status","predictions").count.show()


    model.save("myModelPath")
  }
}
