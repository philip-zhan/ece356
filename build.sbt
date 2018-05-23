name := "ece356lab4"

version := "0.1"

scalaVersion := "2.11.12"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.2.0" withSources(),
  "org.apache.spark" %% "spark-sql" % "2.2.0" withSources(),
  "org.apache.spark" %% "spark-mllib" % "2.2.0" withSources(),
  "mysql" % "mysql-connector-java" % "5.1.45"
)