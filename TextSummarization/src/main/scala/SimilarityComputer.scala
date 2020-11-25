import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.Map
import org.apache.spark.sql.{DataFrame, Row, SQLContext, SQLImplicits, SparkSession}
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.functions._
import com.johnsnowlabs.nlp.{DocumentAssembler, SparkNLP}
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import org.apache.spark.ml.Pipeline

import scala.collection.mutable.ListBuffer
import scala.collection.mutable.WrappedArray
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover}

class SimilarityComputer(var sparkSession: SparkSession, var sc: SparkContext = null, var sentencesDF: DataFrame, var resetProb:Double = 0.15, var iterations: Int)
{
  private object testImplicits extends SQLImplicits {
    protected override def _sqlContext: SQLContext = sparkSession.sqlContext
  }
  import testImplicits._

  def summarize(aggrDF: DataFrame): DataFrame = {
    sc = sparkSession.sparkContext
    val rdd = aggrDF.rdd

    val predictionsArr = rdd
      .collect()
      .map(row => (
        row(0).asInstanceOf[Long],
        summarizeDocument(
          row.getAs[Seq[String]]("sentences").toArray,
          row.getAs[Seq[String]]("sent_ids").toArray)
      ))

    val predictionsDF = sc.parallelize(predictionsArr)
      .toDF("doc_id", "predicted_summary")

    return predictionsDF

  }

  def evaluate (predDF: DataFrame): DataFrame = {
    sc = sparkSession.sparkContext
    val rdd = predDF.rdd
    val performances = rdd
      .collect()
      .map(row => (
        row(0).asInstanceOf[Long],
        getSimilarityForDocument(
          Array(row(1).toString, row(2).toString), // Passing true_summary & pred_summary
          Array("TRUE", "PRED")
        )
      ))

    val resultDF = sc.parallelize(performances)
      .toDF("doc_id", "similarity")
    return resultDF
  }

  def getSimilarityForDocument(sents: Array[String], sentIds: Array[String]):Double = {
    val similarityDF = getSimilarityMatrix(sents, sentIds)
    val similarity = similarityDF.filter(col("sent_id_1") =!= col("sent_id_2"))
      .select("similarity")
      .first()
      .getDouble(0)

    return similarity
  }

  def calculateTFIDF(flattenedKeywordsDF: DataFrame,wordFreqInSentsDF: DataFrame,wordCol: String,sentIdCol: String,numOfSents: Long ): DataFrame = {
    val computeIDF = udf { df: Long =>
      math.abs(math.log((numOfSents.toDouble + 2) / (df.toDouble + 1)))
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

    // Get TF-IDF vectors for all the sentences
    return calculateTFIDF(flattenedKeywordsDF, wordFreqInSentsDF, "word", "sent_id", numSents)
  }

  def getSimilarities (TFIDFVectorsDF: DataFrame) : DataFrame = {
    var result = new ListBuffer[(String, String, Double)]
    val tfidfVectorsArray = TFIDFVectorsDF.rdd
      .map(row => row.toSeq.toArray)
      .collect()

    for (iVector <- tfidfVectorsArray) {
      for (jVector <- tfidfVectorsArray) {
        val iSendId = iVector(0).asInstanceOf[String]
        val jSendId = jVector(0).asInstanceOf[String]
        var sum = 0.0
        var iDist = 0.0
        var jDist = 0.0
        for (index <- iVector.indices) {
          if (index != 0) {
            val i = iVector(index).asInstanceOf[Double]
            val j = jVector(index).asInstanceOf[Double]
            sum = sum + (i * j)
            iDist = iDist + (i*i)
            jDist = jDist + (j*j)
          }
        }
        var similarity = sum / (Math.sqrt(iDist) * Math.sqrt(jDist))
        if (similarity >  1.0) similarity = 1.0
        // Ignore the entry is same sentences are compared.
        if (!iSendId.equals(jSendId)) result = result :+ ((iSendId.toString, jSendId.toString, similarity))
      }
    }
    val resultList = result.toList
    val similarityDF = sc.parallelize(resultList).toDF("sent_id_1", "sent_id_2", "similarity")
    return similarityDF
  }

  def getSimilarityMatrix(sents: Array[String], sentIds: Array[String]) : DataFrame = {
    val transposed = Array(sentIds, sents).transpose
    var sentsDF = sc.parallelize(transposed)
      .map(x => (x(0), x(1)))
      .toDF("sent_id", "sentences")
    var TFIDFVectorsDF = computeTFIDFVectors(sentsDF).select("word", "sent_id", "tf_idf", "squared_tfidf")

    // Get the sparse TF-IDF matrix
    var pivotedTFIDFMatrixDF = TFIDFVectorsDF.groupBy("sent_id").pivot("word").sum("tf_idf").na.fill(0)

    return getSimilarities(pivotedTFIDFMatrixDF)
  }

  def summarizeDocument (sents: Array[String], sentIds: Array[String]) : String = {
    // Step 1: Create a data frame for the sentences in this document
    val transposed = Array(sentIds, sents).transpose
    var sentsDF = sc.parallelize(transposed)
      .map(x => (x(0), x(1)))
      .toDF("sent_id", "sentences")

    // Step 2: Compute the similarity matrix for this document's sentences
    val inputDF = getSimilarityMatrix(sents, sentIds)

    // Step 3: Using this matrix, compute the text ranks and get the final summary
    val summaryForDocument = textRankAlgorithm(inputDF)

    return summaryForDocument
  }

  def textRankAlgorithm (inputDF: DataFrame) : String = {
    var TextRanks = Map[String,Double]()
    val totalNodes=sc.broadcast(inputDF.select($"sent_id_1").distinct().count())

    val sList= inputDF.select($"sent_id_1").distinct().rdd.map(R=>R.get(0)).collect().toList
    sList.foreach(s=>TextRanks.put(s.asInstanceOf[String],1/totalNodes.value.toDouble))

    var TR=sc.broadcast(TextRanks)

    var TextOutWeights=inputDF.groupBy("sent_id_1").sum("similarity").withColumnRenamed("sum(similarity)","sumOutWeights")
    var StagingTable=inputDF.join(TextOutWeights,Seq("sent_id_1"))
      .withColumn("Weight",($"similarity"/$"sumOutWeights"))

    def getRank(Id:String):Double = {
      var ret= TR.value.get(Id).getOrElse(Double).asInstanceOf[Double]
      return ret
    }

    var StagingTableRDD=StagingTable.rdd.map(L=>(L.getAs[String]("sent_id_2"),(L.getAs[String]("sent_id_1"),
      L.getAs[Double]("Weight"))))
    StagingTableRDD.persist()

    var StgRDD2=StagingTableRDD.map{case(x,(y,z))=>(x,getRank(y)*z)}.reduceByKey((a,b)=>a+b)
      .map{case(x,y)=>(x,y*(1-resetProb)+resetProb)}


    TextRanks= collection.mutable.Map()++ StgRDD2.collect().toMap
    TR=sc.broadcast(TextRanks);

    while(iterations>0){
      StgRDD2=StagingTableRDD.map{case(x,(y,z))=>(x,getRank(y)*z)}.reduceByKey((a,b)=>a+b)
        .map{case(x,y)=>(x,y*(1-resetProb)+resetProb)}

      TextRanks= collection.mutable.Map()++ StgRDD2.collect().toMap
      TR=sc.broadcast(TextRanks);
      iterations-=1
    }

    println("\n\n Final Text Ranks:")
    println(TR.value)
    var topSentences=TR.value.toSeq.sortBy(-_._2).map(t=>t._1)
    var returnSize=(totalNodes.value.toDouble * 0.5).round.asInstanceOf[Int]
    println("summarySize: "+returnSize)
    val finalSummarySentenceIds=topSentences.toList.take(returnSize)
    println("\n\nOrdered in desc: "+finalSummarySentenceIds)


    // TODO: Subhajit's logic goes here
//    if (similarityMatrixDF.count() == 0) return "No summary!"
//
//    val someSentID = similarityMatrixDF
//      .orderBy(desc("similarity"))
//      .first()
//      .getString(0)
//
//    val someSentence = sentencesDF
//      .filter(col("sent_id") === someSentID)
//      .first()
//      .getString(1)

    return "someSentence"
  }
}
