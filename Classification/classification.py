'''
DATAAI922 - Big Data Processing
Lab 3 - Part 1 - Classification Task
Author: Thomas Palmeira Ferraz

Master Data AI - Télécom Paris
November, 2021

'''

import pyspark

sc = pyspark.SparkContext('local[4]',appName="Spark Lab Session")
sqlContext = pyspark.SQLContext(sc)

from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SQLContext
from pyspark.ml.feature import StringIndexer

# schema to cast data from file
schema = StructType([StructField("c"+str(i),StringType()) for i in range(23)] + [StructField("label",IntegerType())])

# opening the data
raw_data = sqlContext.read.csv("/content/exo1_train.csv", schema=schema)
raw_data_nn = raw_data.dropna()

# View raw data 
# raw_data_nn.show()

# +-------------+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-----+
# |           c0| c1| c2| c3| c4| c5| c6| c7| c8| c9|c10|c11|c12|c13|c14|c15|c16|c17|c18|c19|c20|c21|c22|label|
# +-------------+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-----+
# | exo1-train-1|  g|  d|  e|  b|  a|  a|  a|  b|  c|  a|  a|  ?|  b|  i|  a|  a|  a|  e|  b|  a|  a|  d|    0|
# | exo1-train-2|  d|  d|  c|  a|  a|  a|  a|  a|  a|  a|  a|  c|  b|  d|  a|  b|  a|  b|  a|  b|  c|  d|    1|
# | exo1-train-3|  e|  d|  d|  c|  a|  a|  a|  b|  e|  d|  c|  c|  a|  f|  b|  a|  a|  e|  b|  b|  c|  d|    0|
# | exo1-train-4|  d|  e|  a|  a|  a|  a|  a|  c|  b|  a|  a|  c|  b|  d|  b|  a|  a|  d|  a|  e|  c|  a|    1|
# | exo1-train-5|  b|  b|  b|  a|  a|  a|  a|  a|  a|  a|  a|  b|  a|  a|  b|  a|  a|  b|  a|  c|  b|  a|    1|
# | exo1-train-6|  d|  e|  a|  a|  a|  a|  a|  b|  c|  a|  a|  c|  b|  e|  b|  a|  a|  d|  a|  d|  b|  d|    1|
# | exo1-train-7|  b|  a|  b|  b|  a|  a|  a|  a|  a|  b|  a|  a|  b|  f|  b|  b|  a|  d|  b|  c|  a|  d|    1|
# | exo1-train-8|  e|  d|  d|  c|  a|  a|  a|  b|  e|  d|  c|  c|  a|  d|  b|  a|  a|  e|  b|  b|  c|  d|    0|
# | exo1-train-9|  d|  d|  d|  c|  a|  a|  a|  e|  c|  d|  c|  c|  a|  c|  b|  a|  a|  e|  b|  d|  b|  d|    0|
# |exo1-train-10|  d|  d|  a|  a|  a|  a|  a|  c|  c|  a|  a|  c|  b|  b|  b|  a|  a|  d|  a|  a|  c|  d|    1|
# +-------------+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-----+
# only showing top 10 rows

# Enconding the features as one hot encoder 
# Obs.: I choose this codification rather than numeric categorization (1, 2, 3, ...) because 
# I do not know if exists any order relation among the features letters (a, b, c). If there is no relation, 
# using numeric categorization can create hard non-linear surfaces, so one-hot encoding solves that and also
# would learn the order relation (if there is one). 

from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer

# Creating a copy to process
data = raw_data_nn.alias('data')

# Store the encoders for predicting new data later
vectorizers = [()]*23

# Encoding each column
for i in range(1,23):
  if i == 7: #all values in c7 are the same, so we drop it
    continue
  
  # create the numerical encoder a -> 0, b -> 1, c -> 2, ...
  string_index = StringIndexer(inputCol="c"+str(i), outputCol="index_c"+str(i))

  # Create the one hot encoder 0 -> [1.0, 0.0, .....], 1 -> [0.0, 1.0, .....]
  onehotencoder = OneHotEncoder(inputCol="index_c"+str(i), outputCol="onehot_c"+str(i), dropLast=False)
  
  # Apply numerical encoder
  df_temp_model = string_index.fit(data)
  df_temp = df_temp_model.transform(data)
  
  # Apply one hot encoder and erase numerical encoder intermediate step
  data_model = onehotencoder.fit(df_temp)
  data = data_model.transform(df_temp).drop("index_c"+str(i))
  
  # Save the encoder model of the column for futher processing
  vectorizers[i] = (df_temp_model,data_model)

# View processed data 
# data.show()

# Sample of the new columns created:
# +-------------+-------------+-------------+
# |    onehot_c1|    onehot_c2|    onehot_c3|
# +-------------+-------------+-------------+
# |(7,[3],[1.0])|(6,[0],[1.0])|(9,[0],[1.0])|
# |(7,[0],[1.0])|(6,[0],[1.0])|(9,[7],[1.0])|
# |(7,[2],[1.0])|(6,[0],[1.0])|(9,[3],[1.0])|
# |(7,[0],[1.0])|(6,[1],[1.0])|(9,[2],[1.0])|
# |(7,[1],[1.0])|(6,[3],[1.0])|(9,[1],[1.0])|
# |(7,[0],[1.0])|(6,[1],[1.0])|(9,[2],[1.0])|
# |(7,[1],[1.0])|(6,[2],[1.0])|(9,[1],[1.0])|
# |(7,[2],[1.0])|(6,[0],[1.0])|(9,[3],[1.0])|
# |(7,[0],[1.0])|(6,[0],[1.0])|(9,[3],[1.0])|
# |(7,[0],[1.0])|(6,[0],[1.0])|(9,[2],[1.0])|
# +-------------+-------------+-------------+
# only showing top 10 rows

# Obs.: the character "?" was also coded as a feature, there is no reason for that but simplicity
# as the model could recognize this is not relevant and them it does not affect the final result. 

# Merge all features into a single vector
assembler = VectorAssembler(inputCols=["onehot_c"+str(i) for i in range(1,22) if i != 7],outputCol="features")
data = assembler.transform(data)
# data.show()

# Sample of the feature column created:
# +--------------------+
# |            features|
# +--------------------+
# |(116,[3,7,13,23,2...|
# |(116,[0,7,20,22,2...|
# |(116,[2,7,16,24,2...|
# |(116,[0,8,15,22,2...|
# |(116,[1,10,14,22,...|
# |(116,[0,8,15,22,2...|
# |(116,[1,9,14,23,2...|
# |(116,[2,7,16,24,2...|
# |(116,[0,7,16,24,2...|
# |(116,[0,7,15,22,2...|
# +--------------------+
# only showing top 10 rows

# Training a model

# Spliting 80% for train and 20% for test
train, test = data.randomSplit([0.8, 0.2], seed=12345)

# Creating the model

from pyspark.ml.classification import DecisionTreeClassifier, DecisionTreeClassificationModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

decisionTree = DecisionTreeClassifier(maxDepth=2,labelCol="label", featuresCol="features")
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",labelCol="label",metricName='areaUnderROC')
evaluator2 = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="label")

# Hyperparameter optmization

from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator

# Parameters searched - Grid Search
paramGrid = ParamGridBuilder()\
    .addGrid(decisionTree.maxBins, range(4,16,4)) \
    .addGrid(decisionTree.maxDepth, range(3,8)) \
    .build()

# Using k-fold cross validation for the search
cvs = CrossValidator(estimator=decisionTree,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator,
                    # 80% of the data will be used for training, 20% for validation.
                    numFolds=5)


cvs_model = cvs.fit(train)
cvs_predictions = cvs_model.transform(test)

# Evaluating over area Under ROC
evaluator.evaluate(cvs_predictions) # Result 1.0

evaluator2.evaluate(cvs_predictions, {evaluator.metricName: "recall"}) # Result  recall = 1.0

evaluator2.evaluate(cvs_predictions, {evaluator.metricName: "precision"}) # Result precision = 1.0
# The f0.2 is then 1.0

# Predicting over the predict dataset

# schema to cast data from file
schema2 = StructType([StructField("c"+str(i),StringType()) for i in range(23)])

# opening the predict data
pred_raw_data = sqlContext.read.csv("/content/exo1_predict.csv",schema=schema2)
pred_raw_data_nn = pred_raw_data.dropna()

# View raw data 
# pred_raw_data_nn.show()

# +--------------+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
# |            c0| c1| c2| c3| c4| c5| c6| c7| c8| c9|c10|c11|c12|c13|c14|c15|c16|c17|c18|c19|c20|c21|c22|
# +--------------+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
# | exo1-graded-1|  b|  a|  b|  a|  a|  a|  a|  a|  a|  c|  a|  d|  a|  e|  b|  a|  a|  b|  a|  b|  b|  a|
# | exo1-graded-2|  d|  d|  b|  a|  a|  a|  a|  a|  a|  a|  a|  c|  b|  e|  a|  b|  a|  c|  a|  b|  a|  d|
# | exo1-graded-3|  d|  d|  b|  a|  a|  a|  a|  a|  a|  a|  a|  c|  b|  g|  b|  a|  a|  d|  a|  d|  c|  a|
# | exo1-graded-4|  b|  a|  b|  b|  a|  a|  a|  a|  a|  b|  b|  a|  b|  a|  b|  b|  a|  d|  b|  a|  c|  a|
# | exo1-graded-5|  e|  d|  d|  c|  a|  a|  a|  e|  c|  d|  c|  c|  a|  f|  b|  a|  a|  e|  b|  d|  b|  a|
# | exo1-graded-6|  e|  d|  e|  b|  a|  a|  a|  a|  a|  d|  c|  ?|  b|  i|  a|  a|  a|  g|  b|  a|  a|  d|
# | exo1-graded-7|  b|  d|  f|  a|  b|  a|  a|  a|  a|  a|  a|  c|  a|  e|  b|  a|  a|  d|  a|  g|  a|  b|
# | exo1-graded-8|  b|  e|  d|  c|  a|  a|  a|  e|  d|  d|  c|  c|  a|  d|  b|  a|  a|  e|  b|  b|  b|  d|
# | exo1-graded-9|  g|  d|  e|  b|  a|  a|  a|  e|  a|  b|  a|  c|  a|  e|  a|  b|  a|  d|  b|  a|  c|  d|
# |exo1-graded-10|  e|  e|  d|  c|  a|  a|  a|  d|  d|  d|  c|  c|  a|  c|  b|  a|  a|  e|  b|  b|  c|  a|
# +--------------+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
# only showing top 10 rows

# Creating a copy to process
pred_data = pred_raw_data_nn.alias('pred_data')

# Encoding each column using the encoders fitted before
for i in range(1,23):
  if i == 7: #all values are the same in this column so we drop it
    continue
  df_temp = vectorizers[i][0].transform(pred_data)
  pred_data = vectorizers[i][1].transform(df_temp).drop("index_c"+str(i))
# Merging the vectors 
pred_data = assembler.transform(pred_data)

# View the data
# pred_data.show()

# Predicting
predicted = cvs_model.transform(pred_data)

# View the data with the predicted values
# predicted.show()

# Gettting only the required fields
answer = predicted.select('c0', 'prediction')

# Checking if the data is in the correct format
# answer.show(5)
# +-------------+----------+
# |           c0|prediction|
# +-------------+----------+
# |exo1-graded-1|       1.0|
# |exo1-graded-2|       1.0|
# |exo1-graded-3|       1.0|
# |exo1-graded-4|       1.0|
# |exo1-graded-5|       0.0|
# +-------------+----------+
# only showing top 5 rows

# Saving the result to submit on moodle
answer.write.csv('exo1')

# Final Remarks

# As the moodle got pretty good results with fast training time, I stoped here. But my plan was to use some 
# kind of dimension reduction (PCA, Chi Squared, SVD) or even try models that do feature selection better 
# (Random Forest and GradBoost). But this was not necessery, Decision Tree is a simple moodle that got good 
# results and the simpler the better. 
