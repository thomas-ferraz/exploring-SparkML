"""
DATAAI922 - Big Data Processing
Lab 3 - Part 2 - Clustering Task
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
schema = StructType([StructField("c0",StringType())] + [StructField("c"+str(i),DoubleType()) for i in range(1, 8)])

# opening the data
raw_data = sqlContext.read.csv("/content/exo2_tocluster.csv",schema=schema)
raw_data_nn = raw_data.dropna()

# View raw data 
# raw_data_nn.show()

# +---------------+-----------+------------+------------+------------+-------------+------------+------------+
# |             c0|         c1|          c2|          c3|          c4|           c5|          c6|          c7|
# +---------------+-----------+------------+------------+------------+-------------+------------+------------+
# | exo2-cluster-1|31002.31324|-80967.66086|  77883.5258| 48843.18235| -99379.82955| -5869.91744| 12089.14713|
# | exo2-cluster-2|30896.98816|-54473.15729| 73882.56923| 76768.40358|-106211.69278|  13313.0134| 16658.48435|
# | exo2-cluster-3|17374.16934|-59403.69201| 27580.31884| 79138.68698| -19277.60536|  3190.04834| 68552.28418|
# | exo2-cluster-4| 30012.5232|-61078.92925| 67626.07322| 82263.71013| -90379.28654| -2586.86963| 13879.08185|
# | exo2-cluster-5|  697.81242|-32725.14544| 49878.31245|  5440.83459|-107648.77193|   610.97418|-46974.20304|
# | exo2-cluster-6|27567.28911|-13115.32517| 56489.30966| 18555.42032|  -86212.6938|-14980.38953|-43984.03257|
# | exo2-cluster-7|12199.71831|-50703.47658| 54011.57375|  9097.74486|  -94281.8466|-19900.40795|-40054.78652|
# | exo2-cluster-8|20507.06204|-30225.49083|   57525.803|   764.16908| -95071.53404|-17295.58658| -41044.9521|
# | exo2-cluster-9| 7800.07608|-74837.87293| 60678.68409| 79398.60973|-101496.79758| 10985.40867|  5056.90568|
# |exo2-cluster-10|18676.33666|-50888.13621| 65104.51364|-42105.88546|-104264.08256|-26216.10278|-41183.32782|
# +---------------+-----------+------------+------------+------------+-------------+------------+------------+
# only showing top 10 rows

# Merging all features into a single vector
assembler=VectorAssembler(inputCols=["c"+str(i) for i in range(1, 7)],outputCol="features")
data=assembler.transform(raw_data_nn)

# View trasnformed data 
# data.show()

# +---------------+-----------+------------+------------+------------+-------------+------------+------------+--------------------+
# |             c0|         c1|          c2|          c3|          c4|           c5|          c6|          c7|            features|
# +---------------+-----------+------------+------------+------------+-------------+------------+------------+--------------------+
# | exo2-cluster-1|31002.31324|-80967.66086|  77883.5258| 48843.18235| -99379.82955| -5869.91744| 12089.14713|[31002.31324,-809...|
# | exo2-cluster-2|30896.98816|-54473.15729| 73882.56923| 76768.40358|-106211.69278|  13313.0134| 16658.48435|[30896.98816,-544...|
# | exo2-cluster-3|17374.16934|-59403.69201| 27580.31884| 79138.68698| -19277.60536|  3190.04834| 68552.28418|[17374.16934,-594...|
# | exo2-cluster-4| 30012.5232|-61078.92925| 67626.07322| 82263.71013| -90379.28654| -2586.86963| 13879.08185|[30012.5232,-6107...|
# | exo2-cluster-5|  697.81242|-32725.14544| 49878.31245|  5440.83459|-107648.77193|   610.97418|-46974.20304|[697.81242,-32725...|
# | exo2-cluster-6|27567.28911|-13115.32517| 56489.30966| 18555.42032|  -86212.6938|-14980.38953|-43984.03257|[27567.28911,-131...|
# | exo2-cluster-7|12199.71831|-50703.47658| 54011.57375|  9097.74486|  -94281.8466|-19900.40795|-40054.78652|[12199.71831,-507...|
# | exo2-cluster-8|20507.06204|-30225.49083|   57525.803|   764.16908| -95071.53404|-17295.58658| -41044.9521|[20507.06204,-302...|
# | exo2-cluster-9| 7800.07608|-74837.87293| 60678.68409| 79398.60973|-101496.79758| 10985.40867|  5056.90568|[7800.07608,-7483...|
# |exo2-cluster-10|18676.33666|-50888.13621| 65104.51364|-42105.88546|-104264.08256|-26216.10278|-41183.32782|[18676.33666,-508...|
# +---------------+-----------+------------+------------+------------+-------------+------------+------------+--------------------+
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
# scaled_data.select("scaled_features").take(2)
# [Row(scaled_features=DenseVector([0.6987, -1.4479, 1.2797, 0.3349, -1.146, 0.6879])),
#  Row(scaled_features=DenseVector([0.692, -0.897, 1.1998, 0.9224, -1.2337, 1.9176]))]

# Training a model - K means

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Silhouette score based on squared euclidian distance - Silhouette Model
silhouette_scores = [] 
# Within Set Sum of Squared Errors - Elbow Method
WSSSE = [] 
# Store the models to easily get the predictions at the end
models = [] 

#Evaluator for the model
evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='scaled_features', metricName='silhouette', distanceMeasure='squaredEuclidean')

# Testing the best k bewteen 2 and 26
for i in range(2,27):
    
    # Create the model with the current k
    kmeans=KMeans(featuresCol='scaled_features', seed=12345, k=i)
    
    # Fit the model on scaled_data
    kmeans_model=kmeans.fit(scaled_data)
    
    # Get the cluster predictions
    predicted=kmeans_model.transform(scaled_data)

    # Obtain the Within Set Sum of Squared Error (WSSSE)
    cost = kmeans_model.summary.trainingCost

    # Obtain the Silhouette Score based on the squared Euclidean Distance
    score=evaluator.evaluate(predicted)

    # Store the scores and the model
    silhouette_scores.append(score)
    WSSSE.append(cost)
    models.append(kmeans_model)
    
    print(str(i)+"-score: ",score, " Elbow cost: ", cost)

# Results:
# 2-score:  0.6940572525386035  Elbow cost:  28185.10066417899
# 3-score:  0.5572529662893303  Elbow cost:  22008.574098344907
# 4-score:  0.5154036406408425  Elbow cost:  19589.86468894736
# 5-score:  0.536897605039987  Elbow cost:  13473.549285828007
# 6-score:  0.45862879712123383  Elbow cost:  17791.69495852098
# 7-score:  0.44294578890199726  Elbow cost:  11272.871057488925
# 8-score:  0.4700434549925121  Elbow cost:  9594.651247561376
# 9-score:  0.48104040376786145  Elbow cost:  9795.01234180191
# 10-score:  0.43899273873641287  Elbow cost:  8758.982152012879
# 11-score:  0.48356969581835235  Elbow cost:  8037.954395677206
# 12-score:  0.45321197963830084  Elbow cost:  7208.48416135576
# 13-score:  0.4465922987243851  Elbow cost:  6893.509580242627
# 14-score:  0.43677026205635366  Elbow cost:  7494.33415775696
# 15-score:  0.42118423144421957  Elbow cost:  7077.149179877458
# 16-score:  0.4409772319229304  Elbow cost:  5890.40624931232
# 17-score:  0.4283320167722974  Elbow cost:  5697.454372608206
# 18-score:  0.4271385997735033  Elbow cost:  5558.662488094609
# 19-score:  0.43170236862463723  Elbow cost:  5235.802158830529
# 20-score:  0.4253866114981317  Elbow cost:  5115.76280587939
# 21-score:  0.4183847107105289  Elbow cost:  4884.837024650021
# 22-score:  0.4092837781562356  Elbow cost:  4818.097487233052
# 23-score:  0.4149247968725953  Elbow cost:  4548.250816931673
# 24-score:  0.3981139093903715  Elbow cost:  4778.18214365885
# 25-score:  0.39830519047512547  Elbow cost:  4336.53584985353
# 26-score:  0.40072040067884984  Elbow cost:  4250.514566821123

# Best k is k = 5: 
# By the Silhouette method we want to maximize the score (ideally getting the global maximum) - k=5 is the best peak
# By the Elbow method we want to minimize Within Set Sum of Squared Error (WSSSE) - k=5 is a local minima and 
# presented the best trade-off between the both methods. 

# Plot the Silhouette score evolution (zipped on moodle)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,27),silhouette_scores)
ax.set_xlabel('k')
ax.set_ylabel('Silhouette score')

# Plot the WSSSE score evolution (zipped on moodle)

fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,27),WSSSE)
ax.set_xlabel('k')
ax.set_ylabel('Within Set Sum of Squared Error')

# Get the predictions for the chosen K, rechecking the score
predicted = models[3].transform(scaled_data) #k = 5
score = evaluator.evaluate(predicted) # score = 0.536897605039987

# View the results
# predicted.show()

# +---------------+-----------+------------+------------+------------+-------------+------------+------------+--------------------+--------------------+----------+
# |             c0|         c1|          c2|          c3|          c4|           c5|          c6|          c7|            features|     scaled_features|prediction|
# +---------------+-----------+------------+------------+------------+-------------+------------+------------+--------------------+--------------------+----------+
# | exo2-cluster-1|31002.31324|-80967.66086|  77883.5258| 48843.18235| -99379.82955| -5869.91744| 12089.14713|[31002.31324,-809...|[0.69869255740314...|         2|
# | exo2-cluster-2|30896.98816|-54473.15729| 73882.56923| 76768.40358|-106211.69278|  13313.0134| 16658.48435|[30896.98816,-544...|[0.69197715598983...|         0|
# | exo2-cluster-3|17374.16934|-59403.69201| 27580.31884| 79138.68698| -19277.60536|  3190.04834| 68552.28418|[17374.16934,-594...|[-0.1702216347991...|         0|
# | exo2-cluster-4| 30012.5232|-61078.92925| 67626.07322| 82263.71013| -90379.28654| -2586.86963| 13879.08185|[30012.5232,-6107...|[0.63558472559807...|         0|
# | exo2-cluster-5|  697.81242|-32725.14544| 49878.31245|  5440.83459|-107648.77193|   610.97418|-46974.20304|[697.81242,-32725...|[-1.2334862516351...|         2|
# | exo2-cluster-6|27567.28911|-13115.32517| 56489.30966| 18555.42032|  -86212.6938|-14980.38953|-43984.03257|[27567.28911,-131...|[0.47967951798884...|         4|
# | exo2-cluster-7|12199.71831|-50703.47658| 54011.57375|  9097.74486|  -94281.8466|-19900.40795|-40054.78652|[12199.71831,-507...|[-0.5001384576486...|         4|
# | exo2-cluster-8|20507.06204|-30225.49083|   57525.803|   764.16908| -95071.53404|-17295.58658| -41044.9521|[20507.06204,-302...|[0.02952786518057...|         4|
# | exo2-cluster-9| 7800.07608|-74837.87293| 60678.68409| 79398.60973|-101496.79758| 10985.40867|  5056.90568|[7800.07608,-7483...|[-0.7806543960119...|         2|
# |exo2-cluster-10|18676.33666|-50888.13621| 65104.51364|-42105.88546|-104264.08256|-26216.10278|-41183.32782|[18676.33666,-508...|[-0.0871970003902...|         4|
# |exo2-cluster-11|19312.13257|-42529.67501| 58901.55097| -7296.33199| -97885.79454|-22468.45354|-36688.32743|[19312.13257,-425...|[-0.0466594118824...|         4|
# |exo2-cluster-12|10012.85806| 70265.45699|-35172.73951|-19715.43182|  43877.14622|-37538.93318|-25987.19941|[10012.85806,7026...|[-0.6395700575444...|         1|
# |exo2-cluster-13|39045.70981|  -47625.447| 32058.98782| 92741.45917|  -7665.14084|  -11734.843| 83124.72156|[39045.70981,-476...|[1.21152992428657...|         0|
# |exo2-cluster-14|18363.58723|-43319.67714| 47748.04539| 14707.86337| -84405.54786|-38633.83398|-26694.27125|[18363.58723,-433...|[-0.1071375308340...|         4|
# |exo2-cluster-15| 9180.15965|  51806.3311|-26569.49161|-34995.58393|  11953.58062|-30423.94682|  2367.77391|[9180.15965,51806...|[-0.6926619144827...|         1|
# |exo2-cluster-16|48582.69322|-13962.67346| 30319.94978| 96118.31879|  15524.00447|-13086.44348| 59319.79893|[48582.69322,-139...|[1.81959660574766...|         0|
# |exo2-cluster-17|29019.83377|-65818.86974| 40894.25222| 49550.61343| -26291.96347| -1783.92941| 72362.84432|[29019.83377,-658...|[0.57229203214653...|         0|
# |exo2-cluster-18| -914.27935| 50206.79322|-33179.86458|-22936.39963|  21269.30308|-25340.27532|-14413.59148|[-914.27935,50206...|[-1.3362712991342...|         1|
# |exo2-cluster-19|22672.42776| 40070.05105|-53817.82283| 38606.75804| 120085.06521|-27163.36987| 89107.79053|[22672.42776,4007...|[0.16758899949369...|         3|
# |exo2-cluster-20|34236.14105|-21479.45415| 23448.51932| 125681.9156|  -1998.02752|  5440.44961| 72813.70635|[34236.14105,-214...|[0.90487755954768...|         0|
# +---------------+-----------+------------+------------+------------+-------------+------------+------------+--------------------+--------------------+----------+
# only showing top 20 rows

# Save the output in the format required by the professor

# Fuction to map numbers into letters
vec = ['a', 'b', 'c', 'd', 'e']
def map_letter(x):
  return vec[x]

# Use the map fuction to map the values in the dataframe (It would be a simple mapValues on Spark RDD)
from pyspark.sql.functions import udf
udf_map = udf(map_letter, StringType())
predicted = predicted.withColumn('output', udf_map(predicted.prediction))

# predicted.show()
# +---------------+-----------+------------+------------+------------+-------------+------------+------------+--------------------+--------------------+----------+------+
# |             c0|         c1|          c2|          c3|          c4|           c5|          c6|          c7|            features|     scaled_features|prediction|output|
# +---------------+-----------+------------+------------+------------+-------------+------------+------------+--------------------+--------------------+----------+------+
# | exo2-cluster-1|31002.31324|-80967.66086|  77883.5258| 48843.18235| -99379.82955| -5869.91744| 12089.14713|[31002.31324,-809...|[0.69869255740314...|         2|     c|
# | exo2-cluster-2|30896.98816|-54473.15729| 73882.56923| 76768.40358|-106211.69278|  13313.0134| 16658.48435|[30896.98816,-544...|[0.69197715598983...|         0|     a|
# | exo2-cluster-3|17374.16934|-59403.69201| 27580.31884| 79138.68698| -19277.60536|  3190.04834| 68552.28418|[17374.16934,-594...|[-0.1702216347991...|         0|     a|
# | exo2-cluster-4| 30012.5232|-61078.92925| 67626.07322| 82263.71013| -90379.28654| -2586.86963| 13879.08185|[30012.5232,-6107...|[0.63558472559807...|         0|     a|
# | exo2-cluster-5|  697.81242|-32725.14544| 49878.31245|  5440.83459|-107648.77193|   610.97418|-46974.20304|[697.81242,-32725...|[-1.2334862516351...|         2|     c|
# | exo2-cluster-6|27567.28911|-13115.32517| 56489.30966| 18555.42032|  -86212.6938|-14980.38953|-43984.03257|[27567.28911,-131...|[0.47967951798884...|         4|     e|
# | exo2-cluster-7|12199.71831|-50703.47658| 54011.57375|  9097.74486|  -94281.8466|-19900.40795|-40054.78652|[12199.71831,-507...|[-0.5001384576486...|         4|     e|
# | exo2-cluster-8|20507.06204|-30225.49083|   57525.803|   764.16908| -95071.53404|-17295.58658| -41044.9521|[20507.06204,-302...|[0.02952786518057...|         4|     e|
# | exo2-cluster-9| 7800.07608|-74837.87293| 60678.68409| 79398.60973|-101496.79758| 10985.40867|  5056.90568|[7800.07608,-7483...|[-0.7806543960119...|         2|     c|
# |exo2-cluster-10|18676.33666|-50888.13621| 65104.51364|-42105.88546|-104264.08256|-26216.10278|-41183.32782|[18676.33666,-508...|[-0.0871970003902...|         4|     e|
# +---------------+-----------+------------+------------+------------+-------------+------------+------------+--------------------+--------------------+----------+------+
# only showing top 10 rows

# Gettting only the required fields
answer = predicted.select('c0', 'output')

# Checking if the data is in the correct format
# answer.show(5)
# +--------------+------+
# |            c0|output|
# +--------------+------+
# |exo2-cluster-1|     c|
# |exo2-cluster-2|     a|
# |exo2-cluster-3|     a|
# |exo2-cluster-4|     a|
# |exo2-cluster-5|     c|
# +--------------+------+
# only showing top 5 rows

# Saving the result to submit on moodle
answer.write.csv('exo2')
