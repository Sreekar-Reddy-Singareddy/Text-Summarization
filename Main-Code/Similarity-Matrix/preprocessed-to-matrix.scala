// Databricks notebook source
import org.apache.spark.sql.{DataFrame,SparkSession}
import org.apache.spark.sql.types.{
    StructType, StructField, StringType, IntegerType}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.WrappedArray
import org.apache.spark.ml.feature.{StopWordsRemover, RegexTokenizer}

// COMMAND ----------

/*
1. Parallelize all the documents so that they are computed independently.
2. Create a mapper function that takes array of sentences and returns a similarity df.
3. Inside the mapper, create a new dataframe with schema as follows:
  - sent_id
  - sentence
4. Pass this dataframe to the tf-idf-computer function which returns a tf-idf matrix.
5. Convert this tf-idf matrix into a pivot table using 'word' as the pivot and grouped by 'sent_id'.
6. Call another function 'get_similarities' that takes this tf-idf matrix and computes the similarity matrix.
7. In 'get_similarities' function, iterate through all the rows in nested manner and compute similarity for two rows (two sentences).
*/
//dbfs:/FileStore/shared_uploads/sxs190008@utdallas.edu/dummy_text_summarization_data.csv

// COMMAND ----------

val sparkSession = SparkSession.builder.getOrCreate()
import sparkSession.implicits._
var mainDF = sparkSession.read
.option("header", "true")
.option("inderSchema", "true")
.csv("dbfs:/FileStore/shared_uploads/sxs190008@utdallas.edu/dummy_text_summarization_data-1.csv")
.toDF()

mainDF.cache()

// COMMAND ----------

val sentsMainDF = mainDF
.withColumn("sent_id_long", monotonically_increasing_id())
.withColumn("sent_id", col("sent_id_long").cast(StringType))
val aggrDFNew = sentsMainDF.groupBy("doc_id").agg(collect_list("sentence") as "sentences", collect_list("sent_id") as "sent_ids")

// COMMAND ----------

def calculateTFIDF(
    flattenedKeywordsDF: DataFrame,
    wordFreqInSentsDF: DataFrame,
    wordCol: String,
    sentIdCol: String,
    numOfSents: Long ) = 
{
  val computeIDF = udf { df: Long =>
    math.log((numOfSents.toDouble + 1) / (df.toDouble + 1))
  }
  
  // Compute TF
  val termFreqDF = flattenedKeywordsDF.groupBy(sentIdCol, wordCol).agg(count(sentIdCol) as "tf")

  // Compute IDF
  val tokenIDF = wordFreqInSentsDF.withColumn("idf", computeIDF(col("df")))

  
  val tfidfDF = termFreqDF
    .join(tokenIDF, Seq(wordCol), "left")
    .withColumn("tf_idf", col("tf") * col("idf"))
    .withColumn(
      "squared_tfidf",
      col("tf") * col("idf") * col("tf") * col("idf")
    )

  tfidfDF
}

// COMMAND ----------

def computeTFIDFVectors (corpusDF: DataFrame) : DataFrame = {
  val numSents = corpusDF.count()
  
  // Splitting senetences into words
  val regexTokenizer = new RegexTokenizer()
  .setPattern("""\W+""")
  .setInputCol("sentences")
  .setOutputCol("words")
  val tokenizedCorpusDF = regexTokenizer.transform(corpusDF)
  
  // Removing stop words from the words array
  val stopWordsRemover = new StopWordsRemover()
  .setInputCol("words")
  .setOutputCol("filtered")
  val pureCorpusDF = stopWordsRemover.transform(tokenizedCorpusDF)
  
  // Exploding the words vector
  val flattenedKeywordsDF = pureCorpusDF
  .withColumn("word", explode(col("filtered")))
  .select("sent_id", "word")
  
  // Computing the document frequency for each word
  val wordFreqInSentsDF = flattenedKeywordsDF.groupBy("word").agg(countDistinct("sent_id") as "df")
  wordFreqInSentsDF.cache()
  
  // Get TF-IDF vectors for all the sentences
  return calculateTFIDF(flattenedKeywordsDF, wordFreqInSentsDF, "word", "sent_id", numSents)
}

// COMMAND ----------

def getSimilarities (TFIDFVectorsDF: DataFrame, magnitudesDF: DataFrame) : DataFrame = {
  var result = new ListBuffer[(String, String, Double)]
  val tfidfVectorsArray = TFIDFVectorsDF.rdd
  .map(row => row.toSeq.toArray)
  .collect()
  val magnitudeArray = magnitudesDF.rdd
  .map(row => (row(0).asInstanceOf[String], row(1).asInstanceOf[Double]))
  .collectAsMap()
  
  for (iVector <- tfidfVectorsArray) {
    for (jVector <- tfidfVectorsArray) {
      val iSendId = iVector(0).asInstanceOf[String]
      val jSendId = jVector(0).asInstanceOf[String]
      val iMag = magnitudeArray(iSendId)
      val jMag = magnitudeArray(jSendId)
      var sum = 0.0
      for (index <- iVector.indices) {
        if (index != 0) sum = sum + (iVector(index).asInstanceOf[Double] * jVector(index).asInstanceOf[Double])
      }
      val similarity = sum / (iMag * jMag)
      result = result :+ ((iSendId.toString, jSendId.toString, similarity))
    }
  }
  val resultList = result.toList
  val similarityDF = sc.parallelize(resultList).toDF("sent_id_1", "sent_id_2", "similarity")
  return similarityDF
}

// COMMAND ----------

def getSimilarityMatrix(sents: Array[String], sentIds: Array[String]) : DataFrame = {
  val transposed = Array(sentIds, sents).transpose
  var sentsDF = sc.parallelize(transposed)
  .map(x => (x(0), x(1)))
  .toDF("sent_id", "sentences")
  var TFIDFVectorsDF = computeTFIDFVectors(sentsDF).select("word", "sent_id", "tf_idf", "squared_tfidf")
  
  // Get magnitudes of each document
  val sqrtUdf = udf (
    (c: Double) => {Math.sqrt(c)}
  )
  val magnitudesDF = TFIDFVectorsDF
  .select("sent_id", "squared_tfidf")
  .groupBy("sent_id")
  .agg(sum("squared_tfidf") as "temp_magnitude")
  .withColumn("magnitude", sqrtUdf(col("temp_magnitude")))
  
  // Get the sparse TF-IDF matrix
  var pivotedTFIDFMatrixDF = TFIDFVectorsDF.groupBy("sent_id").pivot("word").sum("tf_idf").na.fill(0)
  
  return getSimilarities(pivotedTFIDFMatrixDF, magnitudesDF)
}

// COMMAND ----------

val rdd = aggrDFNew.rdd
// rdd.collect()
// val result = rdd.collect().map(row => getSimilarityMatrix(row(1).asInstanceOf[Array[String]], row(2).asInstanceOf[Array[String]]))
val result = rdd.collect().map(row => getSimilarityMatrix(row.getAs[Seq[String]]("sentences").toArray, row.getAs[Seq[String]]("sent_ids").toArray))

// COMMAND ----------

// display(result(0))
display(result(1))
// display(result(2))

// COMMAND ----------

// 
