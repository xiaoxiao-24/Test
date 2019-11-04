package neumann

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}


object imsi_binaryclassification extends App{

  val conf = new SparkConf().setAppName("SparkIMSI").setMaster("local[*]")
  val sc = new SparkContext(conf)

  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark = SparkSession.builder().appName("IMSI").getOrCreate()

  val path = "data_source/imsi_json.json"

  val imsiDF = spark.read.json(path)

  imsiDF.createOrReplaceTempView("imsiDF")

  implicit def bool2int(b:Boolean) = if (b) 1 else 0
  spark.udf.register("bool2int", bool2int _)

  val imsi_train = spark.sql("""select arpu,averagemonthlybill, totalspent,
                              bool2int(smartphone), handsettype, category,type, daysactive,
                              dayslastactive, bool2int(canceled)
                              from imsiDF
                              where canceled is not null
                              and category <> ''
                              and type <> ''
                              and handsettype <> ''  """)

  // one-hot encoding on our categorical features
  // First we have to use the StringIndexer to convert the strings to integers.
  val indexer1 = new StringIndexer().
    setInputCol("handsettype").
    setOutputCol("handsettypeIndex").
    setHandleInvalid("keep")
  val DF_Churn1 = indexer1.fit(imsi_train).transform(imsi_train)

  val indexer2 = new StringIndexer().
    setInputCol("category").
    setOutputCol("categoryIndex").
    setHandleInvalid("keep")
  val DF_Churn2 = indexer2.fit(DF_Churn1).transform(DF_Churn1)

  val indexer3 = new StringIndexer().
    setInputCol("type").
    setOutputCol("typeIndex").
    setHandleInvalid("keep")
  val DF_Churn3 = indexer3.fit(DF_Churn2).transform(DF_Churn2)

  // encode the indexed columns
  // use the OneHotEncoderEstimator to do the encoding.
  val encoder = new OneHotEncoderEstimator().
    setInputCols(Array("handsettypeIndex", "categoryIndex","typeIndex")).
    setOutputCols(Array("handsettypeVec", "categoryVec","typeVec"))
  val DF_Churn_encoded = encoder.fit(DF_Churn3).transform(DF_Churn3)

  // Spark models need exactly two columns: “label” and “features”
  // create label column
  val get_label = (DF_Churn_encoded.select(DF_Churn_encoded.col("UDF:bool2int(canceled)").as("label"),
    DF_Churn_encoded.col("arpu"), DF_Churn_encoded.col("averagemonthlybill"),
    DF_Churn_encoded.col("totalspent"), DF_Churn_encoded.col("UDF:bool2int(smartphone)"),
    DF_Churn_encoded.col("daysactive"), DF_Churn_encoded.col("dayslastactive"),
    DF_Churn_encoded.col("handsettypeVec"), DF_Churn_encoded.col("categoryVec"),
    DF_Churn_encoded.col("typeVec")))

  // assembler tous les features
  val assembler = new VectorAssembler().setInputCols(Array("arpu",
    "averagemonthlybill", "totalspent", "UDF:bool2int(smartphone)", "daysactive",
    "dayslastactive", "handsettypeVec", "categoryVec", "typeVec")).
    setOutputCol("features")

  // Transform the DataFrame
  val output = assembler.transform(get_label).select("label","features")

  // prepare dataset ( one part for train 70%, one for test 30%)
  // Splitting the data by create an array of the training and test data
  val Array(training, test) = output.select("label","features").
    randomSplit(Array(0.7, 0.3), seed = 12345)

  // create the training model
  val rf = new RandomForestClassifier()

  // create the param grid
  val paramGrid = new ParamGridBuilder().
    addGrid(rf.numTrees,Array(20,50,100)).
    build()

  // create cross val object, define scoring metric
  val cv = new CrossValidator().
    setEstimator(rf).
    setEvaluator(new MulticlassClassificationEvaluator().setMetricName("weightedRecall")).
    setEstimatorParamMaps(paramGrid).
    setNumFolds(3).
    setParallelism(2)

  // train the model
  // You can then treat this object as the model and use fit on it.
  val model = cv.fit(training)

  // get the results of training, test with the test dataset
  val results = model.transform(test).select("features", "label", "prediction")

  println(results)
  results.printSchema()
  results.show(10, false)

  import spark.implicits._
  // convert these results(get "prediction" and "label" to compare) to an RDD
  val predictionAndLabels = results.
    select("prediction","label").
    as[(Double, Double)].
    rdd

  // create our metrics objects and print out the confusion matrix.
  // Instantiate a new metrics objects
  val bMetrics = new BinaryClassificationMetrics(predictionAndLabels)
  val mMetrics = new MulticlassMetrics(predictionAndLabels)
  val labels = mMetrics.labels

  // Print out the Confusion matrix
  println("Confusion matrix:")
  println(mMetrics.confusionMatrix)

  // ---------------------------
  // print metrics
  // ---------------------------

  // Precision by label
  labels.foreach { l =>
    println(s"Precision($l) = " + mMetrics.precision(l))
  }

  // Recall by label
  labels.foreach { l =>
    println(s"Recall($l) = " + mMetrics.recall(l))
  }

  // False Positive Rate by label
  // ROC curve, X line: FPR = FP/N
  labels.foreach { l =>
    println(s"FPR($l) = " + mMetrics.falsePositiveRate(l))
  }

  // F-measure by label
  labels.foreach { l =>
    println(s"F1-Score($l) = " + mMetrics.fMeasure(l))
  }

  // Precision by threshold
  val precision = bMetrics.precisionByThreshold
  precision.foreach { case (t, p) =>
    println(s"Threshold: $t, Precision: $p")
  }


  // Precision-Recall Curve
  val PRC = bMetrics.pr

  // AUPRC
  val auPRC = bMetrics.areaUnderPR
  println("Area under precision-recall curve = " + auPRC)

  // Compute thresholds used in ROC and PR curves
  val thresholds = precision.map(_._1)
  thresholds.collect()

  // ROC Curve
  val roc = bMetrics.roc

  // AUROC
  val auROC = bMetrics.areaUnderROC
  println("Area under ROC = " + auROC)


  spark.stop()
  sc.stop()

}
