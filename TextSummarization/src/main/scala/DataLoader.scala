import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.Map
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import com.johnsnowlabs.nlp.{DocumentAssembler, SparkNLP}
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import org.apache.spark.ml.Pipeline

import scala.collection.mutable.ListBuffer
import scala.collection.mutable.WrappedArray
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover}

object DataLoader {

  def loadRawData(inputFilePath: String, sparkSession: SparkSession, recordsSize: Int) :DataFrame = {
    val mainDF = sparkSession.read
      .option("header", "true")
      .option("multiLine", "true")
      .option("escape", "\"")
      .option("delimiter", ",")
      .option("inferSchema", "true")
      .csv(inputFilePath)
      .toDF("headline", "title", "text")
      .select(col("*"), lower(col("text")))
      .drop("text")
      .withColumnRenamed("lower(text)", "text")
      .limit(recordsSize)

    return mainDF
  }

  def loadSentencesDataFor(rawDF: DataFrame) : DataFrame = {
    // Creating document annotator
    val docAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")
    val documentDF = docAssembler.transform(rawDF)

    // Creating sentence annotator and splitting docs into sentences
    val sentenceDetector = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentences")
    val sentencesDF = sentenceDetector.transform(documentDF)

    // Exploding the array of sentences into sentence column
    val resDF = sentencesDF
      .select("doc_id", "text", "sentences.result")
      .withColumn("sentence", explode(col("result"))).select("doc_id", "sentence")
      .withColumn("sent_id_long", monotonically_increasing_id())
      .withColumn("sent_id", col("sent_id_long").cast(StringType))
      .drop("sent_id_long")
      .drop("text")

    return resDF
  }
}
