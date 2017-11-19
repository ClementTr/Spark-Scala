package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.feature.{CountVectorizer, IDF, RegexTokenizer, StopWordsRemover}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.explode


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
   /*
   * D'abord on commence par charger l'ensemble des données nettoyées auparavant.
   * Spark étant distribué, il faut chargé l'ensemble des fichiers .parquet
   *
   */
   val df_cleaned = spark
     .read
     .load("data/train_cleaned/*.parquet")
    println(df_cleaned.count)
    df_cleaned.select("text").show(5)


    /** Tokenizer **/
    /*
    * Ici l'objectif est de segmenter la colonne text en une liste
    * de mots.
    */
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")
    val df_tokenized = tokenizer.transform(df_cleaned)


    /** Stopwords + Hapax **/
    /*
    * Hapax, partie personelle. Ce sont les mots qui n'apparaissent qu'une seule
    * fois dans l'ensemble des données. On les supprime comme les stopwords car ils
    * n'ont pas d'impact sur le score.
    */
    val df_hapax = df_tokenized
      .select(explode($"tokens").as("value"))
      .groupBy("value")
      .count
      .filter($"count" <= 1)
    val hapax_any = df_hapax
      .select("value")
      .rdd.map(r => r(0))
      .collect()
    val hapax_string : Array[String] =
      (hapax_any map (_.toString)).toArray
    val stopwords = StopWordsRemover.
      loadDefaultStopWords("english")
    val hapax_stopwords = hapax_string ++ stopwords

    val hapax_remover = new StopWordsRemover()
      .setStopWords(hapax_string)
      .setInputCol("tokens")
      .setOutputCol("without_stop_hapax")
    val df_without_hapax = hapax_remover.transform(df_tokenized)


    /** CountVectorizer **/
      /*
      * Le CountVectorizer va nous permettre de transformer nos données texte
      * en données numériques. Plus facile pour l'apprentissage.
      */
    val vectorizer = new CountVectorizer()
      .setInputCol("without_stop_hapax")
      .setOutputCol("vectorize")
    val df_vectorized = vectorizer
      .fit(df_without_hapax)
      .transform(df_without_hapax)
    df_vectorized.show(5)


    /** TFIDF */
      /*
      * La partie IDF nous permet de mettre des poids sur l'occurence des mots,
      * justement pour négliger ceux qui appraissent trop souvent par exemple.
      */
    val idf = new IDF()
      .setInputCol("vectorize")
      .setOutputCol("tfidf")
    val df_tfidf = idf
      .fit(df_vectorized)
      .transform(df_vectorized)
    df_tfidf.select("text", "without_stop_hapax", "tfidf").show(5)


    /** StringIndexer **/
      /*
      * Pour les monnaies ou les pays, le string indexer nous permet de
      * catégorsier des données textuelles en données numériques.
      * Encore une fois c'est pour l'apprentissage futur que nous faisons
      * cette étape.
      */
    val indexer_country = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
    val df_country_indexed =
      indexer_country
        .fit(df_tfidf)
        .transform(df_tfidf)

    val indexer_currency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
    val df_country_currency_indexed =
      indexer_currency
        .fit(df_country_indexed)
        .transform(df_country_indexed)


    /** Vector Assembler **/
      /*
      * Pour les modèles d'apprentissage, Spark préfère travailler avec un vecteur
      * plutôt qu'avec plusieurs colonnes.
      */
    val assembler = new VectorAssembler()
      .setInputCols(Array(
        "tfidf", "days_campaign", "hours_prepa", "goal",
        "country_indexed", "currency_indexed"))
      .setOutputCol("features")
    val df_assembled = assembler.transform(df_country_currency_indexed)


    /** Regression logistique **/
      /*
      * Pour faire notre categorisation, on commence par faire une regression
      * logistique. En argument il nous faut mettre la colonne créée juste auparavant features.
      * Notre target est la colonne features. Nos données sont comme demandé entrainée
      * et testée sur un ration 70%/30%.
      */
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


    /** Pipeline **/
      /*
      * Notre pipeline va permettre d'automatiser l'ensemble des tâches créées
      * auparavant.
      */
    val pipeline = new Pipeline()
      .setStages(Array(
        tokenizer,
        hapax_remover,
        vectorizer,
        idf,
        indexer_country,
        indexer_currency,
        assembler,
        lr
      ))


    /** Avant l'apprentissage **/
      /*
      * Ici, on selectionne les colonnes features qui nous interessent.
      * Pour la réussite du modèle, on cast notre labele en double
      */
    val df_init = df_cleaned
      .withColumn("final_status", 'final_status.cast("Double"))
    print("lol10")
    val Array(training, test) = df_init
      .select("text", "goal", "country2", "currency2",
        "deadline2", "launched_at2", "days_campaign",
        "created_at2", "hours_prepa", "final_status")
      .randomSplit(Array(0.9, 0.1))


    /** ParamGrid **/
      /*
      * Idéal pour tester plusieurs paramètres à notre modèle et trouver
      * les paramètres optimaux
      */
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(
        scala.math.pow(10,-8),
        scala.math.pow(10,-6),
        scala.math.pow(10,-4),
        scala.math.pow(10,-2)))
      .addGrid(vectorizer.minDF, Array(55.0, 75.0, 85.0, 95.0)).build()


    /** F1 Score **/
      /*
      * Pour de la catégorisation, la métrique F1 Score est très appropiée
      * car elle prend en compte la precision et le recall.
      */
    val eval = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")


    /** TrainValidationSplit **/
      /*
      * On suit les instruction du TP qui nous demande d'utiliser en chaque point
      * de la grille, 70 des données pour l'entrainement et 30% pour la validation.
      * On
      */
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(eval)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    /** APPRENTISSAGE ET EVALUATION **/
      /*
      * Désormais on fit nos données selon le modèle choisi en passant par
      * la pipeline définie.
      * Après on pourra regarder le score de notre modèle selon le F1 Score.
      * Et sauver ce modèle
      */
    val model = trainValidationSplit.fit(training)
    val df_WithPredictions = model.transform(test)

    val score = eval.evaluate(df_WithPredictions)
    print("My score: ", score)
    df_WithPredictions.groupBy("final_status","predictions").count.show()


    /** SAUVEGARDE DU MODELE **/
    /*
    * Ajout de overwrite pour pouvoir écraser notre modèle s'il est déjà existant
    */
    model.write.overwrite().save("myModelPath")
    print("Saved !")

    /** FIN **/
  }
}
