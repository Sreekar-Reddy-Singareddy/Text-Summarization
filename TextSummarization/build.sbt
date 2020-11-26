name := "BD_Project"

version := "0.1"

scalaVersion := "2.11.12"

lazy val root = (project in file(".")).
  settings(
    name := "BD_Project",
    version := "1.0",
    scalaVersion := "2.11.12",
    mainClass in Compile := Some("BD_Project")
  )
resolvers += Resolver.url("Typesafe Ivy releases", url("https://repo.typesafe.com/typesafe/ivy-releases"))(Resolver.ivyStylePatterns)

val sparkVersion = "2.4.0"
val johnSnowLabsVersion = "2.6.3"

libraryDependencies ++= Seq(
  "org.scala-lang" % "scala-reflect" % "2.11.12",
  "org.scala-lang" % "scala-library"  % "2.11.12",
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-streaming" % sparkVersion,
  "org.apache.spark" %% "spark-hive" % sparkVersion,
  "com.johnsnowlabs.nlp" %% "spark-nlp" % johnSnowLabsVersion,
  "com.typesafe" % "config" % "1.3.4",
  "com.google.code.gson" % "gson" % "2.7",
)

scalacOptions := Seq("-target:jvm-1.8")
javacOptions ++= Seq("-source", "1.8", "-target", "1.8", "-Xlint")

initialize := {
  val _ = initialize.value
  val javaVersion = sys.props("java.specification.version")
  if (javaVersion != "1.8")
    sys.error("Java 1.8 is required for this project. Found " + javaVersion + " instead")
}

//assemblyMergeStrategy in assembly := {
//  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
//  case x => MergeStrategy.first
//}
