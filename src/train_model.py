from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, hour, dayofweek
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
# Nouvel import pour définir le schéma manuellement
from pyspark.sql.types import StructType, StructField, DoubleType, TimestampType

# 1. Session Spark
spark = SparkSession.builder \
    .appName("Taxi_Train_Tuning") \
    .master("local[*]") \
    .getOrCreate()

# 2. Chargement des données (CORRECTION ROBUSTE)
print("Chargement des données avec schéma strict...")

# On définit explicitement ce qu'on veut lire.
# Cela permet d'ignorer 'VendorID' qui cause le conflit (int vs bigint).
schema = StructType([
    StructField("tpep_pickup_datetime", TimestampType(), True),
    StructField("trip_distance", DoubleType(), True),
    StructField("total_amount", DoubleType(), True)
])

# On force l'utilisation de ce schéma
df = spark.read.schema(schema).parquet("/app/data/yellow_tripdata_2023-*.parquet")

# 3. Feature Engineering
print("Préparation des colonnes...")
# Pas besoin de cast ici car le schema l'a déjà fait !
df = df.withColumn("hour", hour(col("tpep_pickup_datetime"))) \
       .withColumn("day_of_week", dayofweek(col("tpep_pickup_datetime")))

# 4. Nettoyage
# On garde les courses réalistes
df = df.filter((col("trip_distance") > 0.5) & (col("trip_distance") < 200)) \
       .filter((col("total_amount") > 2.5) & (col("total_amount") < 500))

# Échantillonnage
# On prend 10% des données pour l'exercice
df_sample = df.sample(withReplacement=False, fraction=0.1, seed=42)
print(f"Entraînement sur environ {df_sample.count()} lignes avec Cross-Validation...")

# 5. Définition du Pipeline
# A. VectorAssembler
assembler = VectorAssembler(
    inputCols=["trip_distance", "hour", "day_of_week"],
    outputCol="features",
    handleInvalid="skip"
)

# B. LinearRegression
lr = LinearRegression(featuresCol="features", labelCol="total_amount")

# C. Pipeline
pipeline = Pipeline(stages=[assembler, lr])

# 6. CONFIGURATION DU TUNING (CROSS-VALIDATION)

# A. La Grille de Paramètres
print("Configuration de la grille de recherche...")
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5]) \
    .build()

# B. L'Evaluateur (RMSE)
evaluator = RegressionEvaluator(labelCol="total_amount", predictionCol="prediction", metricName="rmse")

# C. Le CrossValidator
cv = CrossValidator(estimator=pipeline,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator,
                    numFolds=3)

# 7. Entraînement
print("Lancement de l'entraînement et de la recherche du meilleur modèle...")
print("Cela peut prendre 2 à 5 minutes selon ta machine. Patience...")

cvModel = cv.fit(df_sample)

# 8. Résultat et Sauvegarde
best_model = cvModel.bestModel
print("✅ Meilleur modèle trouvé !")

best_lr = best_model.stages[-1]
print(f"   -> Meilleur regParam: {best_lr.getRegParam()}")
print(f"   -> Meilleur elasticNetParam: {best_lr.getElasticNetParam()}")

print("Sauvegarde du Pipeline Optimisé...")
best_model.write().overwrite().save("/app/data/pipeline_final")
print("🎉 Terminé ! Le modèle est prêt.")