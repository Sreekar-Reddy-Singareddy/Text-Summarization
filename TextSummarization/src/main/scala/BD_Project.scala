import com.johnsnowlabs.nlp.annotators.Normalizer
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.Map
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import com.johnsnowlabs.nlp.{DocumentAssembler, SparkNLP}
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import org.apache.spark.ml.Pipeline

import scala.collection.mutable.ListBuffer
import scala.collection.mutable.WrappedArray
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover}

object BD_Project {

  def main(args: Array[String]): Unit = {
    if (args.length != 5) {
      println("Please provide input as follows: <raw data path> <output folder path> <iterations of algorithm> <reset prob> <number of records>")
      return
    }
    val inputFilePath = args(0)
    val outputFilePath = args(1)
    val iterationsOfTextRanking = args(2).toInt
    val resetProb = args(3).toDouble
    val recordsToConsider = args(4).toInt

    val sparkSession = SparkSession.builder()
      .appName("BD Project")
      .master("local[4]")
      .getOrCreate()

//     Contains data as it is from the CSV along with an "doc_id" column.
    val rawDF = DataLoader.loadRawData(inputFilePath, sparkSession, recordsToConsider)
      .withColumn("doc_id", monotonically_increasing_id())
    rawDF.cache()

    // Contains all other info of the document such as "headline", "title", "doc_id"
    val documentInfoRDD = rawDF.select("doc_id", "headline").rdd
      .map(row => (row(0).asInstanceOf[Long], row(1).asInstanceOf[String]))
    documentInfoRDD.persist()

    // Contains only "doc_id", "sentence"
    val sentenceDF = DataLoader.loadSentencesDataFor(rawDF)
    sentenceDF.cache()

    // Contains "doc_id", "sentences", "sent_ids"
    val aggrDF = sentenceDF
      .groupBy("doc_id")
      .agg(collect_list("sentence") as "sentences", collect_list("sent_id") as "sent_ids")
    val similarityComputer = new SimilarityComputer(sparkSession = sparkSession, sentencesDF = sentenceDF, resetProb = resetProb, iterations = iterationsOfTextRanking)
    val predictionsDF = similarityComputer.summarize(aggrDF) // id, [sent-ids]

    val predsDF = predictionsDF
      .withColumn("sent_id", explode(col("predicted_sent_ids")))
      .drop("predicted_sent_ids")
    val df1 = sentenceDF.as("T1")
    val df2 = predsDF.as("T2")
    val joinedRDD = df1.join(df2, col("T1.sent_id") === col("T2.sent_id"))
      .select(col("T1.doc_id"), col("T1.sentence")).rdd
      .map(row => (row(0).asInstanceOf[Long], row(1).asInstanceOf[String]))
    val finalSummaryRDD = joinedRDD
      .reduceByKey((x,y) => x+". "+y)
      .join(documentInfoRDD)

    finalSummaryRDD
      .coalesce(1, shuffle = true)
      .saveAsTextFile(outputFilePath+"/final-summaries-"+System.nanoTime().toString)

    val predictionsRDD = finalSummaryRDD
      .map(doc => (
        doc._1,
        doc._2._1.toLowerCase().split("""\W+"""),
        doc._2._2.toLowerCase().split("""\W+""")))
      .map(d => (d._1, d._2.toSet, d._3.toSet, d._3.length))
      .map(d => (d._1, d._2, d._3.filter(x => x.length>3), d._4, (d._2.intersect(d._3).size.toDouble/d._4.toDouble)))
      .map(d => (d._1, d._5))

    predictionsRDD
      .coalesce(1, true)
      .saveAsTextFile(outputFilePath+"/final-rogue-predictions-"+System.nanoTime().toString)
  }
}