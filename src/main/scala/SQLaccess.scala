import java.util.Properties
import org.apache.spark.ml.classification.{DecisionTreeClassifier, LogisticRegression}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

object SQLaccess extends App {
  Class.forName("com.mysql.jdbc.Driver").newInstance
  val spark = SparkSession
    .builder()
    .master("local")
    .appName("ece356lab4")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")
  import spark.implicits._
//  val sparkConf = new SparkConf().setMaster("local").setAppName("testSparkLocal")
//  val sc = new SparkContext(sparkConf)
//  sc.setLogLevel("ERROR")
//  val sqlContext = new SQLContext(sc)

  val connectionProperties = new Properties()
  connectionProperties.put("user", "philipzhan")
  connectionProperties.put("password", "Ycwgy1996")

//  val user_table = spark.read.jdbc("jdbc:mysql://localhost:3306/yelp_db", "user", connectionProperties)
//  val review_table = spark.read.jdbc("jdbc:mysql://localhost:3306/yelp_db", "review", connectionProperties)
//  val business_table = spark.read.jdbc("jdbc:mysql://localhost:3306/yelp_db", "business", connectionProperties)
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


//  user_review_business_0.printSchema()
//  user_review_business_0.show(10)

//  val schema = StructType(
//    Seq(
//      StructField("label", IntegerType),
//      StructField("features", )
//    )
//  )

  val libsvm = user_review_business_5.map{
    row =>
      (
        row.getAs[Int]("review_stars").toDouble - 1,
        Vectors.dense(
//          row.getAs[Int]("review_useful").toDouble,
//          row.getAs[Int]("review_funny").toDouble,
//          row.getAs[Int]("review_cool").toDouble,
          row.getAs[Int]("user_review_count").toDouble,
          row.getAs[Int]("user_useful").toDouble,
          row.getAs[Int]("user_funny").toDouble,
          row.getAs[Int]("user_cool").toDouble,
          row.getAs[Int]("user_fans").toDouble,
          row.getAs[java.math.BigDecimal]("true_avg_stars").doubleValue(),
//          row.getAs[Double]("user_avg_stars"),
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
          //          row.getAs[Int]("review_useful").toDouble,
          //          row.getAs[Int]("review_funny").toDouble,
          //          row.getAs[Int]("review_cool").toDouble,
          row.getAs[Int]("user_review_count").toDouble,
          row.getAs[Int]("user_useful").toDouble,
          row.getAs[Int]("user_funny").toDouble,
          row.getAs[Int]("user_cool").toDouble,
          row.getAs[Int]("user_fans").toDouble,
          row.getAs[java.math.BigDecimal]("true_avg_stars").doubleValue(),
          //          row.getAs[Double]("user_avg_stars"),
          row.getAs[Double]("business_latitude"),
          row.getAs[Double]("business_longitude"),
          row.getAs[Int]("business_review_count").toDouble,
          row.getAs[Int]("business_is_open").toDouble,
          row.getAs[Double]("business_stars")
        )
      )
  }.toDF("label", "features")

//  libsvm.printSchema()
////  println(libsvm.count())
//  libsvm.show(false)

  // Load training data
//  val training = spark
//    .read
//    .format("libsvm")
//    .load("/Users/philipzhan/GitHub/spark/data/mllib/sample_multiclass_classification_data.txt")
//  training.printSchema()
//  training.show(false)

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
  println("Logistic Regression")
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
  println("Decision Tree")
  prediction_decision_tree.select($"label" + 1, $"features", $"prediction" + 1).sample(false, 0.02).show(10, false)
  println($"Decision Tree Accuracy: ${accuracy_decision_tree}")
  println(s"Decision Tree Number of classes: ${classes_dt}")
  println(s"Decision Tree Number of features: ${features_dt}")

  println($"Training set size: ${trainingSize}")
  println($"Test set size: ${testSize}")


  //  val paramGrid_lr = new ParamGridBuilder()
//    .addGrid(logisticRegression.regParam, Array(0, 0.1))
//    .addGrid(logisticRegression.elasticNetParam, Array(0.5, 1.0))
//    .build()
//  val paramGrid_dt = new ParamGridBuilder().build()
//
//  val cv_lr = new CrossValidator()
//    .setEstimator(logisticRegression)
//    .setEstimatorParamMaps(paramGrid_lr)
//    .setEvaluator(new MulticlassClassificationEvaluator().setMetricName("accuracy"))
//    .setNumFolds(5)  // Use 3+ in practice
//  val cvModel_lr = cv_lr.fit(libsvm)
//
//  val cv_dt = new CrossValidator()
//    .setEstimator(decisionTree)
//    .setEstimatorParamMaps(paramGrid_dt)
//    .setEvaluator(new MulticlassClassificationEvaluator().setMetricName("accuracy"))
//    .setNumFolds(5)  // Use 3+ in practice
//  val cvModel_dt = cv_dt.fit(libsvm)


  //  println(predict.count())






//  println(s"getMaxIter: ${logisticRegression.getMaxIter}")
//  println(s"getRegParam: ${logisticRegression.getRegParam}")
//  println(s"getElasticNetParam: ${logisticRegression.getElasticNetParam}")

//  println("Decision Tree")
//  println($"numClasses: ${decisionTree.numClasses}")
//  println($"numFeatures: ${decisionTree.numFeatures}")
//  println($"rootNode: ${decisionTree.rootNode}")


//  val success = predict.filter("label = prediction").count()
//  val accuracy = 100.0 * success / predict.count()
//  // Print the coefficients and intercept for multinomial logistic regression
//  println(s"Coefficients: \n${model.coefficientMatrix}")
//  println(s"Intercepts: ${model.interceptVector}")
//  println(s"numClasses: ${model.numClasses}")
//  println(s"numFeatures: ${model.numFeatures}")
//  println(s"getMaxIter: ${model.getMaxIter}")
//  println(s"getRegParam: ${model.getRegParam}")
//  println(s"getElasticNetParam: ${model.getElasticNetParam}")
////  println($"success: $success")
//  println($"accuracy: $accuracy%")

}
