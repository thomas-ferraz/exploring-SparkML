"""
DATAAI922 - Big Data Processing
Lab 3 - Part 3 - Regression Task
Author: Thomas Palmeira Ferraz

Master Data AI - Télécom Paris
November, 2021

"""

import pyspark
sc = pyspark.SparkContext('local[4]',appName="Spark Lab Session")
sqlContext = pyspark.SQLContext(sc)

from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SQLContext
from pyspark.ml.feature import StringIndexer

# schema to cast data from file
schema = StructType([StructField("c0",StringType())] + [StructField("c"+str(i),IntegerType()) for i in range(1, 8)] + [StructField("target",DoubleType())])

# opening the data
raw_data = sqlContext.read.csv("/content/exo3_train.csv",schema=schema)
raw_data_nn = raw_data.dropna()

# View raw data 
# raw_data_nn.show()

# +-------------+---+-----+----+----+----+----+-----+------+
# |           c0| c1|   c2|  c3|  c4|  c5|  c6|   c7|target|
# +-------------+---+-----+----+----+----+----+-----+------+
# | exo3-train-1|330|-1579|  60|-294| 960| 116|-1027|  34.0|
# | exo3-train-2|586| -852| 158| 340| 523| -40| -615| 553.0|
# | exo3-train-3|379|-1347| 221|-133| 881| 199|-1049| 518.0|
# | exo3-train-4|626| -834| 183| 390| 470|-112| -466| 233.0|
# | exo3-train-5|765| -769| 231| 247| 445|-152| -498| 717.0|
# | exo3-train-6|534|  326|-180|-227|-404|-543|  515| 310.0|
# | exo3-train-7|651|  111|-388|-469|-175|-409|  325| 757.0|
# | exo3-train-8|609|   82|-312|-533|-187|-423|  182|1049.0|
# | exo3-train-9|377|-1448| 301|-236| 765| -56| -908|  57.0|
# |exo3-train-10|636|   34|-297|-363|-204|-445|  371| 387.0|
# +-------------+---+-----+----+----+----+----+-----+------+
# only showing top 10 rows

# Merging all features into a single vector
assembler=VectorAssembler(inputCols=["c"+str(i) for i in range(1, 8)],outputCol="features")
data=assembler.transform(raw_data_nn)

# View trasnformed data 
# data.show()

# +-------------+---+-----+----+----+----+----+-----+------+--------------------+
# |           c0| c1|   c2|  c3|  c4|  c5|  c6|   c7|target|            features|
# +-------------+---+-----+----+----+----+----+-----+------+--------------------+
# | exo3-train-1|330|-1579|  60|-294| 960| 116|-1027|  34.0|[330.0,-1579.0,60...|
# | exo3-train-2|586| -852| 158| 340| 523| -40| -615| 553.0|[586.0,-852.0,158...|
# | exo3-train-3|379|-1347| 221|-133| 881| 199|-1049| 518.0|[379.0,-1347.0,22...|
# | exo3-train-4|626| -834| 183| 390| 470|-112| -466| 233.0|[626.0,-834.0,183...|
# | exo3-train-5|765| -769| 231| 247| 445|-152| -498| 717.0|[765.0,-769.0,231...|
# | exo3-train-6|534|  326|-180|-227|-404|-543|  515| 310.0|[534.0,326.0,-180...|
# | exo3-train-7|651|  111|-388|-469|-175|-409|  325| 757.0|[651.0,111.0,-388...|
# | exo3-train-8|609|   82|-312|-533|-187|-423|  182|1049.0|[609.0,82.0,-312....|
# | exo3-train-9|377|-1448| 301|-236| 765| -56| -908|  57.0|[377.0,-1448.0,30...|
# |exo3-train-10|636|   34|-297|-363|-204|-445|  371| 387.0|[636.0,34.0,-297....|
# +-------------+---+-----+----+----+----+----+-----+------+--------------------+
# only showing top 10 rows

# Scalling the data such that for all columns the mean is zero and the standard deviation is one. 

from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)

# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(data)

# Normalize each feature to have unit standard deviation.
scaled_data = scalerModel.transform(data)

# View scaled data
# scaled_data.show()

# Checking first two lines of this column
scaled_data.select("scaled_features").take(2)
# [Row(scaled_features=DenseVector([-0.768, -1.2853, 0.0268, -1.2011, 1.2961, 1.0522, -1.1941])),
#  Row(scaled_features=DenseVector([0.3223, -0.198, 0.4387, 1.8105, 0.3457, 0.4216, -0.4164]))]

# Training a model - Gradient-boosted trees (GBTs)
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Spliting 80% for train and 20% for test
train, test = scaled_data.randomSplit([0.8, 0.2], seed=12345)

evaluator = RegressionEvaluator(labelCol="target", predictionCol="prediction", metricName="rmse")
gbt = GBTRegressor(featuresCol="scaled_features", labelCol="target", maxIter=10, lossType='squared')


# Hyperparameter optmization

from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator

# Parameters searched - Grid Search
paramGrid = ParamGridBuilder()\
    .addGrid(gbt.maxBins, range(4,32,4)) \
    .addGrid(gbt.maxDepth, range(3,8)) \
    .build()

# Using k-fold cross validation for the search
cvs = CrossValidator(estimator=gbt,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator,
                    # 80% of the data will be used for training, 20% for validation.
                    numFolds=5)

cvs_model = cvs.fit(train)


#Evaluating on test set with RMSE 
cvs_predictions = cvs_model.transform(test)
evaluator.evaluate(cvs_predictions) # RMSE = 152.1898281871939

# Getting the winner model
# bestModel = cvs_model.bestModel
# bestModel._java_obj.getMaxBins() # 28
# bestModel._java_obj.getMaxDepth() # 7
# print(bestModel) # GBTRegressionModel: uid=GBTRegressor_f800e1ba8f31, numTrees=10, numFeatures=7


# Training another model - Random Forest Regressor

from pyspark.ml.regression import RandomForestRegressor

rf = RandomForestRegressor(featuresCol="scaled_features", labelCol="target")

# Parameters searched - Grid Search
paramGrid2 = ParamGridBuilder()\
    .addGrid(rf.maxBins, range(4,32,4)) \
    .addGrid(rf.maxDepth, range(3,8)) \
    .build()

# Using k-fold cross validation for the search
cvs2 = CrossValidator(estimator=rf,
                    estimatorParamMaps=paramGrid2,
                    evaluator=evaluator,
                    # 80% of the data will be used for training, 20% for validation.
                    numFolds=5)

cvs2_model = cvs2.fit(train)

#Evaluating on test set with RMSE 
cvs2_predictions = cvs2_model.transform(test)
evaluator.evaluate(cvs2_predictions) # RMSE = 164.8356055863281


# Training another model - Linear Regression

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol="scaled_features", labelCol="target", maxIter=10000)

# Parameters searched - Grid Search
# regParam = 0 means no regularization and if elasticNetParam = 0 
# it neglets ElasticNet and to the standard linear regression
paramGrid3 = ParamGridBuilder()\
    .addGrid(lr.regParam, [0,0.1,0.3,0.5,0.7,0.9,1.0]) \
    .addGrid(lr.elasticNetParam, [0,0.1,0.3,0.5,0.7,0.9,1.0]) \
    .build()

# Using k-fold cross validation for the search
cvs3 = CrossValidator(estimator=lr,
                    estimatorParamMaps=paramGrid3,
                    evaluator=evaluator,
                    # 80% of the data will be used for training, 20% for validation.
                    numFolds=5)

cvs3_model = cvs3.fit(train)

#Evaluating on test set with RMSE 
cvs3_predictions = cvs3_model.transform(test)
evaluator.evaluate(cvs3_predictions) #RMSE = 81.69280563396606

# Getting the winner model
# bestModel = cvs3_model.bestModel
# bestModel._java_obj.getRegParam() # 0.0
# bestModel._java_obj.getElasticNetParam() # 0.0

# Using the last model (Linear Regression) to annotate the predict dataset

# schema to cast data
schema2 = StructType([StructField("c0",StringType())] + [StructField("c"+str(i),IntegerType()) for i in range(1, 8)])

# opening the predict data
pred_raw_data = sqlContext.read.csv("/content/exo3_predict.csv",schema=schema2)
pred_raw_data_nn = pred_raw_data.dropna()

# View raw data 
# pred_raw_data_nn.show()

# +--------------+---+-----+----+----+----+----+-----+
# |            c0| c1|   c2|  c3|  c4|  c5|  c6|   c7|
# +--------------+---+-----+----+----+----+----+-----+
# | exo3-graded-1|132|-1417| 338| 189| 851| 259|-1081|
# | exo3-graded-2|405|-1410| 387|-196| 743| -36| -947|
# | exo3-graded-3|204|-1663| 321| -95| 945| 210|-1280|
# | exo3-graded-4|287|-1355| 450| 109| 699|  25| -847|
# | exo3-graded-5|637|   26|-319|-554|-169|-434|  188|
# | exo3-graded-6|493|   -5|-228|-245|-148|-290|  126|
# | exo3-graded-7|798| -969|   3|  14| 537|-248| -415|
# | exo3-graded-8|197|-1563| 173|  19|1024| 374|-1235|
# | exo3-graded-9|481|  402| -31|  79|-333|-239|  269|
# |exo3-graded-10|300|-1480| 263| -99| 800|  13| -885|
# +--------------+---+-----+----+----+----+----+-----+
# only showing top 10 rows

# Merging all features into a single vector and calling the data using the mean 
# and std we calculated on train dataset

pred_data=assembler.transform(pred_raw_data_nn)
pred_scaled_data = scalerModel.transform(pred_data)

# View Predict Scaled Data
# pred_scaled_data.show()

# +--------------+---+-----+----+----+----+----+-----+--------------------+--------------------+
# |            c0| c1|   c2|  c3|  c4|  c5|  c6|   c7|            features|     scaled_features|
# +--------------+---+-----+----+----+----+----+-----+--------------------+--------------------+
# | exo3-graded-1|132|-1417| 338| 189| 851| 259|-1081|[132.0,-1417.0,33...|[-1.6111713627660...|
# | exo3-graded-2|405|-1410| 387|-196| 743| -36| -947|[405.0,-1410.0,38...|[-0.4485610401897...|
# | exo3-graded-3|204|-1663| 321| -95| 945| 210|-1280|[204.0,-1663.0,32...|[-1.3045488601085...|
# | exo3-graded-4|287|-1355| 450| 109| 699|  25| -847|[287.0,-1355.0,45...|[-0.9510812528784...|
# | exo3-graded-5|637|   26|-319|-554|-169|-434|  188|[637.0,26.0,-319....|[0.53944480170661...|
# | exo3-graded-6|493|   -5|-228|-245|-148|-290|  126|[493.0,-5.0,-228....|[-0.0738002036083...|
# | exo3-graded-7|798| -969|   3|  14| 537|-248| -415|[798.0,-969.0,3.0...|[1.22508678681572...|
# | exo3-graded-8|197|-1563| 173|  19|1024| 374|-1235|[197.0,-1563.0,17...|[-1.3343593812002...|
# | exo3-graded-9|481|  402| -31|  79|-333|-239|  269|[481.0,402.0,-31....|[-0.1249039540512...|
# |exo3-graded-10|300|-1480| 263| -99| 800|  13| -885|[300.0,-1480.0,26...|[-0.8957188565652...|
# +--------------+---+-----+----+----+----+----+-----+--------------------+--------------------+
# only showing top 10 rows

# Predicting

predicted = cvs3_model.transform(pred_scaled_data)

# View the data with the predicted values
# predicted.show()

# +--------------+---+-----+----+----+----+----+-----+--------------------+--------------------+-------------------+
# |            c0| c1|   c2|  c3|  c4|  c5|  c6|   c7|            features|     scaled_features|         prediction|
# +--------------+---+-----+----+----+----+----+-----+--------------------+--------------------+-------------------+
# | exo3-graded-1|132|-1417| 338| 189| 851| 259|-1081|[132.0,-1417.0,33...|[-1.6111713627660...| 23.892645095958756|
# | exo3-graded-2|405|-1410| 387|-196| 743| -36| -947|[405.0,-1410.0,38...|[-0.4485610401897...| 289.95676582917304|
# | exo3-graded-3|204|-1663| 321| -95| 945| 210|-1280|[204.0,-1663.0,32...|[-1.3045488601085...| 344.17873402518376|
# | exo3-graded-4|287|-1355| 450| 109| 699|  25| -847|[287.0,-1355.0,45...|[-0.9510812528784...|-131.28912146784018|
# | exo3-graded-5|637|   26|-319|-554|-169|-434|  188|[637.0,26.0,-319....|[0.53944480170661...|  889.8244561570796|
# | exo3-graded-6|493|   -5|-228|-245|-148|-290|  126|[493.0,-5.0,-228....|[-0.0738002036083...|  744.9671156142888|
# | exo3-graded-7|798| -969|   3|  14| 537|-248| -415|[798.0,-969.0,3.0...|[1.22508678681572...|  152.5875289539107|
# | exo3-graded-8|197|-1563| 173|  19|1024| 374|-1235|[197.0,-1563.0,17...|[-1.3343593812002...| 326.40351036384953|
# | exo3-graded-9|481|  402| -31|  79|-333|-239|  269|[481.0,402.0,-31....|[-0.1249039540512...| 1057.7503190441685|
# |exo3-graded-10|300|-1480| 263| -99| 800|  13| -885|[300.0,-1480.0,26...|[-0.8957188565652...|-288.67796708899243|
# +--------------+---+-----+----+----+----+----+-----+--------------------+--------------------+-------------------+
# only showing top 10 rows


# Gettting only the required fields
answer = predicted.select('c0', 'prediction')

# Checking if the data is in the correct format
# answer.show(5)
# +-------------+-------------------+
# |           c0|         prediction|
# +-------------+-------------------+
# |exo3-graded-1| 23.892645095958756|
# |exo3-graded-2| 289.95676582917304|
# |exo3-graded-3| 344.17873402518376|
# |exo3-graded-4|-131.28912146784018|
# |exo3-graded-5|  889.8244561570796|
# +-------------+-------------------+
# only showing top 5 rows


# Saving the result to submit on moodle
answer.write.csv('exo3')

# Final Remarks

# As I do not know the data well, I tried at first to use models that do their own feature selection
# (Gradient-boosted trees and Random Forest), but errors were high in both models.  Then I tried
# Linear regression and got a tolerable RMSE error (about 82). The reason for this seems to be
# that the data has a linear profile type, in which case Tree-based methods might fail and,
# obviously, linear regression would work well. An RMSE of 82 is tolerable considering the range
# of output values.
