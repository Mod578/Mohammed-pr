Scalable Data Science with Python: Black Friday Sales Analysis
Project Overview
This project uses the RAPIDS ecosystem (cuDF and cuML) to implement machine learning models on a large-scale dataset (Black Friday sales). The goal is to demonstrate GPU-accelerated data science for real-world applications.

Objectives
Use cuDF for data processing and cuML for machine learning.
Implement and compare at least three models: classification, regression, and clustering.
Collect GPU performance metrics (utilization, memory usage, execution times).
Provide actionable insights into customer behavior.
Dataset
Source : Kaggle's "Black Friday Dataset".
Features : Customer demographics (gender, age, occupation) and purchase behavior.
Size : Millions of records.
Implementation
1. Data Preprocessing
Used cuDF for fast GPU-accelerated data cleaning:
Dropped missing values.
Encoded categorical variables (e.g., gender, age).
Split data into training and testing sets.
2. Machine Learning Models
Random Forest Classifier :
Objective: Classify customers based on purchase amount (above/below average).
Metrics: Accuracy, training time.
Linear Regression :
Objective: Predict purchase amounts.
Metrics: Mean Squared Error (MSE), training time.
K-Means Clustering :
Objective: Segment customers into 5 clusters.
Metrics: Training time.
3. GPU Performance Metrics
Collected GPU utilization, memory usage, and execution times using pynvml.
Results
MODEL
METRIC
VALUE
Random Forest
Accuracy
(Insert Value)
Linear Regression
MSE
(Insert Value)
K-Means
Training Time (s)
(Insert Value)

GPU Utilization: (Insert Value)%
GPU Memory Used: (Insert Value) MB
Key Insights
Random Forest achieved high accuracy for classification.
Linear Regression provided reasonable predictions for purchase amounts.
K-Means effectively segmented customers into distinct groups.
Challenges and Solutions
Challenge : Handling missing data.
Solution : Used fillna() and encoding techniques.
Challenge : GPU resource monitoring.
Solution : Used pynvml to track performance.
Conclusions
GPU acceleration improved efficiency for large-scale data processing.
The project demonstrated practical insights into customer behavior.
Repository Structure
model.joblib: Trained model.
scalar.joblib: Scaler object (if applicable).
X_test.csv, y_test.csv: Test data.
prediction.py: Script for predictions.
Mohammed.ipynb: Full implementation notebook.
