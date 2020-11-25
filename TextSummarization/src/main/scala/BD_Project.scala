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

object BD_Project {

  def main(args: Array[String]): Unit = {
    val sparkSession = SparkSession.builder()
      .appName("BD Project")
      .master("local[1]")
      .getOrCreate()

    val inputFilePath = args(0)
    val dataLoader  = new DataLoader()

    // Contains data as it is from the CSV along with an "doc_id" column.
    val rawDF = dataLoader.loadRawData(inputFilePath, sparkSession)
      .withColumn("doc_id", monotonically_increasing_id())

    // Contains all other info of the document such as "headline", "title", "doc_id"
    val documentInfoDF = rawDF.select("doc_id", "headline", "title")

    // Contains only "doc_id", "sentence"
    val sentenceDF = dataLoader.loadSentencesDataFor(rawDF)

//    rawDF.select("doc_id","text").show()
//    documentInfoDF.select("doc_id", "headline").show()
//    sentenceDF.select("doc_id","sentence")show()


    // Contains "doc_id", "sentences", "sent_ids"
    val aggrDF = sentenceDF
      .groupBy("doc_id")
      .agg(collect_list("sentence") as "sentences", collect_list("sent_id") as "sent_ids")
    val similarityComputer = new SimilarityComputer(sparkSession = sparkSession, sentencesDF = sentenceDF, resetProb = 0.2, iterations = 5)
    val predictionsDF = similarityComputer.summarize(aggrDF)
    predictionsDF.show()

    // Join real summary with predicted summary
    val df1 = documentInfoDF.as("real")
    val df2 = predictionsDF.as("pred")
    val joinedPredictionsDF = df1.join(df2, col("real.doc_id") === col("pred.doc_id"))
      .select(
        col("real.doc_id").alias("doc_id"),
        col("real.headline").alias("true_summary"),
        col("pred.predicted_summary").alias("pred_summary")
      )
    joinedPredictionsDF.show()

    // Do the performance evaluation between true and predicted summaries here
    val performanceEvaluationDF = similarityComputer.evaluate(joinedPredictionsDF)
    performanceEvaluationDF.show()

  }
}