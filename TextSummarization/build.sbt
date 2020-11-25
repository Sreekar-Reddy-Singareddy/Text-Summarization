name := "BD_Project"

version := "0.1"

scalaVersion := "2.11.8"

val sparkVersion = "2.2.1"
val johnSnowLabsVersion = "2.6.3"


libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-streaming" % sparkVersion,
  "org.apache.spark" %% "spark-hive" % sparkVersion,
  "com.johnsnowlabs.nlp" %% "spark-nlp" % johnSnowLabsVersion
)
