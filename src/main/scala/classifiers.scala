import java.util.Properties
import org.apache.spark.ml.classification.{DecisionTreeClassifier, LogisticRegression}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

object classifiers {
  Class.forName("com.mysql.jdbc.Driver").newInstance
  val spark = SparkSession
    .builder()
    .master("local")
    .appName("ece356lab4")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")
  import spark.implicits._

  val connectionProperties = new Properties()
  connectionProperties.put("user", "philipzhan")
  connectionProperties.put("password", "password")

  val user_review_business_5 = spark.read.jdbc(
    "jdbc:mysql://localhost:3306/yelp_db",
    "user_review_business_5",
    connectionProperties
  )
  val testTable = spark.read.jdbc(
    "jdbc:mysql://localhost:3306/yelp_db",
    "user_review_business_9",
    connectionProperties
  )

  val libsvm = user_review_business_5.map{
    row =>
      (
        row.getAs[Int]("review_stars").toDouble - 1,
        Vectors.dense(
          row.getAs[Int]("user_review_count").toDouble,
          row.getAs[Int]("user_useful").toDouble,
          row.getAs[Int]("user_funny").toDouble,
          row.getAs[Int]("user_cool").toDouble,
          row.getAs[Int]("user_fans").toDouble,
          row.getAs[java.math.BigDecimal]("true_avg_stars").doubleValue(),
          row.getAs[Double]("business_latitude"),
          row.getAs[Double]("business_longitude"),
          row.getAs[Int]("business_review_count").toDouble,
          row.getAs[Int]("business_is_open").toDouble,
          row.getAs[Double]("business_stars")
        )
      )
  }.toDF("label", "features")

  val test = testTable.map{
    row =>
      (
        row.getAs[Int]("review_stars").toDouble - 1,
        Vectors.dense(
          row.getAs[Int]("user_review_count").toDouble,
          row.getAs[Int]("user_useful").toDouble,
          row.getAs[Int]("user_funny").toDouble,
          row.getAs[Int]("user_cool").toDouble,
          row.getAs[Int]("user_fans").toDouble,
          row.getAs[java.math.BigDecimal]("true_avg_stars").doubleValue(),
          row.getAs[Double]("business_latitude"),
          row.getAs[Double]("business_longitude"),
          row.getAs[Int]("business_review_count").toDouble,
          row.getAs[Int]("business_is_open").toDouble,
          row.getAs[Double]("business_stars")
        )
      )
  }.toDF("label", "features")

  val Array(trainingData, testData) = libsvm.randomSplit(Array(0.5, 0.5))
  val trainingSize = trainingData.count().toString
  val testSize = test.count().toString
  val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

  // Logistic Regression
  val logisticRegression = new LogisticRegression()
    .setMaxIter(200)
    .setRegParam(0)
    .setElasticNetParam(1)
  val model_lr = logisticRegression.fit(trainingData)
  val prediction_logistic_regression = model_lr.transform(test)
  val accuracy_logistic_regression = evaluator.evaluate(prediction_logistic_regression).toString
  val classes_lr = model_lr.numClasses.toString
  val features_lr = model_lr.numFeatures.toString
  prediction_logistic_regression.select($"label" + 1, $"features", $"prediction" + 1).sample(false, 0.02).show(10, false)
  println($"Logistic Regression Accuracy: ${accuracy_logistic_regression}")
  println(s"Logistic Regression Number of classes: ${classes_lr}")
  println(s"Logistic Regression Number of features: ${features_lr}")

  // Decision Tree
  val decisionTree = new DecisionTreeClassifier()
  val model_dt = decisionTree.fit(trainingData)
  val prediction_decision_tree = model_dt.transform(test)
  val accuracy_decision_tree = evaluator.evaluate(prediction_decision_tree).toString
  val classes_dt = model_dt.numClasses.toString
  val features_dt = model_dt.numFeatures.toString
  prediction_decision_tree.select($"label" + 1, $"features", $"prediction" + 1).sample(false, 0.02).show(10, false)
  println($"Decision Tree Accuracy: ${accuracy_decision_tree}")
  println(s"Decision Tree Number of classes: ${classes_dt}")
  println(s"Decision Tree Number of features: ${features_dt}")
  println($"Training set size: ${trainingSize}")
  println($"Test set size: ${testSize}")
}
