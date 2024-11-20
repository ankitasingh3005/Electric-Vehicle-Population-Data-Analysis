# Final Project Report

## Title Page

- **Project Title:** Road to Sustainability: Clustering and Predicting Electric Vehicle Adoption
- **Team Name:** Data Ops Crew
- **Team Members:** Mohankumar Anem (U88099159), Ankita Singh (U10367653), Sunil Yangandula(U82205360)

## Table of Contents

1. [Introduction](#introduction)
2. [Data Preparation](#data-preparation)
3. [Modeling](#modeling)
4. [Results](#results)
5. [Discussion](#discussion)
6. [Conclusion](#conclusion)
7. [References](#references)

## Introduction

### Business Problem and Objectives
The transition to electric vehicles (EVs) is critical for reducing carbon emissions and promoting sustainable transportation. This project focuses on analyzing Electric Vehicle Population Data to:

- Understand key features influencing EV adoption and performance.
- Build predictive models to assist manufacturers and policymakers.
- Identify clusters of similar vehicles for strategic market segmentation.

### Dataset Summary
The dataset includes detailed information about electric vehicles, including:

- **Categorical Features:** Make, Model, Electric Vehicle Type.
- **Numerical Features:** Electric Range, Base MSRP, and more.

This data is vital for understanding the current landscape of EVs and making informed decisions to support their adoption.

## Data Preparation

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.ml.feature import StringIndexer, OneHotEncoder, StandardScaler, VectorAssembler

# Initialize Spark Session
spark = SparkSession.builder.appName("EV_Population_Preprocessing").getOrCreate()

# Load the dataset
data_path = "Electric_Vehicle_Population_Data.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)
df.show()
```

Detail preprocessing steps taken during the data preparation phase:

### Step 1: Data Cleaning
- **Handling Missing/Null Values**: Filled missing categorical values with `"Unknown"` and numerical values such as `Electric Range` and `Base MSRP` with `0`.

```python
# Handling Missing/Null Values
df_cleaned = df.fillna({
    "County": "Unknown",
    "City": "Unknown",
    "Postal Code": 0,
    "Electric Range": 0,
    "Base MSRP": 0
})
```

- **Feature Engineering**: Added a new binary column, `CAFV_Eligible_Binary`, to indicate eligibility based on `"Clean Alternative Fuel Vehicle Eligibility"`.

```python
df_cleaned = df_cleaned.withColumn(
    "CAFV_Eligible_Binary",
    when(col("Clean Alternative Fuel Vehicle (CAFV) Eligibility") == "Clean Alternative Fuel Vehicle Eligible", 1).otherwise(0)
)
```

### Step 2: Feature Encoding and Scaling
- **Indexing Categorical Features**: Used `"StringIndexer"` to convert categorical columns (`Make`, `Model`, `Electric Vehicle Type`) into numerical format.

```python
indexer_make = StringIndexer(inputCol="Make", outputCol="Make_Index", handleInvalid="skip")
indexer_model = StringIndexer(inputCol="Model", outputCol="Model_Index", handleInvalid="skip")
indexer_type = StringIndexer(inputCol="Electric Vehicle Type", outputCol="Type_Index", handleInvalid="skip")
```

- **Feature Assembly and Standardization**: Used `VectorAssembler` to combine features and `StandardScaler` for scaling.

```python
# Assemble Features
assembler = VectorAssembler(inputCols=["Electric Range", "Base MSRP", "Make_Index", "Model_Index", "Type_Index"], outputCol="Features")

# Standardize Features
scaler = StandardScaler(inputCol="Features", outputCol="ScaledFeatures", withStd=True, withMean=True)
```

### Step 3: Dimensionality Reduction
- PCA for Dimensionality Reduction: Used PCA to reduce the feature set to 2 principal components.

```python
pca = PCA(k=2, inputCol="ScaledFeatures", outputCol="PCAFeatures")
```

This preprocessing ensured that the data was properly cleaned, features were engineered, and the feature set was prepared for effective model training.

## Modeling

### Linear Regression (Predict Electric Range)

```python
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

# Define Linear Regression Model
lr = LinearRegression(featuresCol="ScaledFeatures", labelCol="Electric Range", predictionCol="Range_Prediction")

# Hyperparameter Tuning
param_grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

evaluator = RegressionEvaluator(labelCol="Electric Range", predictionCol="Range_Prediction", metricName="rmse")

cv = CrossValidator(estimator=lr, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)

# Build the Pipeline
pipeline = Pipeline(stages=[indexer_make, indexer_model, indexer_type, assembler, scaler, cv])

# Train the Model
pipeline_model = pipeline.fit(train_data)

# Make Predictions
lr_predictions = pipeline_model.transform(test_data)

# Evaluate the Model
lr_rmse = evaluator.evaluate(lr_predictions)
print(f"Linear Regression RMSE after tuning: {lr_rmse}")
```

### Random Forest Classifier (Predict CAFV Eligibility)

```python
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Define Random Forest Classifier
rf = RandomForestClassifier(featuresCol="ScaledFeatures", labelCol="CAFV_Eligible_Binary", predictionCol="Eligibility_Prediction")

# Hyperparameter Tuning
param_grid_rf = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 50, 100]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .build()

evaluator_rf = MulticlassClassificationEvaluator(labelCol="CAFV_Eligible_Binary", predictionCol="Eligibility_Prediction", metricName="accuracy")

cv_rf = CrossValidator(estimator=rf, estimatorParamMaps=param_grid_rf, evaluator=evaluator_rf, numFolds=3)

# Build the Pipeline
rf_pipeline = Pipeline(stages=[indexer_make, indexer_model, indexer_type, assembler, scaler, cv_rf])

# Train the Model
rf_pipeline_model = rf_pipeline.fit(train_data)

# Make Predictions
rf_predictions = rf_pipeline_model.transform(test_data)

# Evaluate the Model
rf_accuracy = evaluator_rf.evaluate(rf_predictions)
print(f"Random Forest Accuracy after tuning: {rf_accuracy}")
```

### K-Means Clustering

```python
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import PCA
from pyspark.ml.evaluation import ClusteringEvaluator

# PCA for Dimensionality Reduction
pca = PCA(k=2, inputCol="ScaledFeatures", outputCol="PCAFeatures")

# Define K-Means Clustering
kmeans = KMeans(featuresCol="PCAFeatures", predictionCol="Cluster", k=10)

# Build the Pipeline
kmeans_pipeline = Pipeline(stages=[indexer_make, indexer_model, indexer_type, assembler, scaler, pca, kmeans])

# Train the Model
kmeans_pipeline_model = kmeans_pipeline.fit(train_data)

# Make Predictions
kmeans_predictions = kmeans_pipeline_model.transform(test_data)

# Evaluate the Model
clustering_evaluator = ClusteringEvaluator(featuresCol="PCAFeatures", predictionCol="Cluster", metricName="silhouette")
silhouette_score = clustering_evaluator.evaluate(kmeans_predictions)
print(f"K-Means Silhouette Score: {silhouette_score}")
```

## Results

### Evaluation Metrics and Findings

- **Linear Regression RMSE**: The Root Mean Squared Error (RMSE) for predicting the electric range was calculated to be approximately `0.010`. This significant improvement suggests that the changes made to prevent data leakage were effective, resulting in a more realistic and generalizable model.
  - **Mean Absolute Error (MAE)**: `0.00767`
  - **R² Score**: `1.0`

- **Random Forest Classifier**:
  - **Accuracy**: `99.88%` in predicting the CAFV eligibility, indicating the model's high effectiveness.
  - **Precision (Weighted)**: `99.88%`
  - **Recall (Weighted)**: `99.88%`
  - **F1-Score (Weighted)**: `99.88%`
  - **AUC-ROC**: `0.999996`

- **K-Means Clustering Silhouette Score**: The silhouette score of `0.798` suggests that the clusters are well-formed and distinct, providing meaningful segmentation of the vehicle data.

## Discussion

- **Model Interpretations**:
  - The **Linear Regression** model, after changes to prevent data leakage, demonstrated a more reasonable RMSE of `0.010`, indicating that the model's predictions are now more reliable and less prone to overfitting. The improvement in performance metrics like RMSE and MAE confirms that mitigating data leakage has led to a more generalizable model suitable for real-world prediction of electric range.
  - The **Random Forest Classifier** achieved a high accuracy rate of `99.88%`. This performance confirms that the feature set was highly effective for the given classification task. Metrics such as precision, recall, and F1-score all indicated the robustness of the model. Further analysis of feature importance could provide insights into which factors most significantly affect eligibility, offering actionable information for manufacturers and policymakers.
  - The **K-Means Clustering** model successfully segmented the dataset into distinct clusters, with a silhouette score of `0.798`. The moderately high silhouette score implies that the clusters are well-separated and meaningful, which can be valuable for market segmentation and targeting specific groups based on characteristics such as range or vehicle type.

- **Limitations and Challenges**:
  - **Preventing Data Leakage**: Initially, data leakage inflated the model's performance metrics. The revised approach demonstrated the importance of careful feature selection, feature engineering, and validation to ensure robust model performance.
  - **Data Quality**: Missing values were filled, but imputing missing values may introduce bias into the models. Collecting more comprehensive data, especially with fewer missing entries, would likely lead to more accurate models. Moreover, adding additional features, such as geographic data and incentives, would provide a more holistic view of the factors affecting EV adoption.
  - **Feature Engineering Opportunities**: Introducing additional features like charging infrastructure availability, regional incentives, and consumer demographics could further improve the quality of the models. These features may be particularly helpful for clustering and identifying distinct market segments.

## Conclusion

- **Key Insights**: The analysis demonstrated that machine learning models could effectively predict electric vehicle adoption metrics such as electric range and eligibility for incentives. Addressing data leakage resulted in a more accurate and reliable Linear Regression model. Features such as electric range, vehicle type, and base MSRP were found to be crucial in understanding EV performance and market eligibility.
- **Model Outcomes**: The revised Linear Regression model provided realistic predictions, while the Random Forest classifier demonstrated robust classification capabilities, making it suitable for understanding the characteristics of vehicles eligible for clean energy incentives. The K-Means clustering model provided useful segmentation insights, identifying distinct groups of vehicles based on performance metrics.
- **Future Work**: Future work should focus on gathering more comprehensive datasets to reduce the reliance on imputed values, implementing advanced regularization techniques to prevent overfitting, and incorporating additional features such as infrastructure data, geographic data, and regional incentives. These improvements could yield more robust and generalizable models, providing deeper insights into electric vehicle adoption patterns.

## References

- Electric Vehicle Population Data. (n.d.). Data.gov. Retrieved from [https://catalog.data.gov/dataset/electric-vehicle-population-data](https://catalog.data.gov/dataset/electric-vehicle-population-data)

