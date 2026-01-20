# Machine Learning Zoomcamp (Cohort 2025)

## Capstone Project 2: A Stroke Predictor Service

### Project Description ###

Stroke is a medical condition in which poor blood flow to the brain causes cell death. According to the World Health Organization (WHO), stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths. Early identification of high-risk individuals is critical for preventative healthcare. However, many risk factors—such as hypertension, heart disease, and lifestyle choices—interact in complex ways that are difficult for clinicians to quantify manually in a rapid screening environment.

#### Problem Statement

Medical professionals often lack automated, data-driven tools to assist in the triage of patients who are at high risk of stroke. While clinical data is frequently collected (e.g., BMI, glucose levels, smoking status), it is often stored without being leveraged for predictive analytics. There is a need for a reliable, scalable system that can process patient demographics and health indicators to provide a probability of stroke occurrence, allowing doctors to prioritize diagnostic tests and lifestyle interventions.

#### Objective

The objective of this project is to develop an end-to-end Machine Learning pipeline to predict whether a patient is likely to get a stroke based on input parameters such as gender, age, various diseases, and smoking status.

##### Key Goals:

* **Analyze and Prepare:** Perform extensive EDA on the "Stroke Prediction Dataset" to identify key drivers of stroke risk.

* **Model Development:** Train and tune multiple classification models (including linear models like Logistic Regression and tree-based models like Random Forest/XGBoost) to find the most accurate predictor.

* **Productionalize:** Package the best-performing model into a production-ready web service using Flask and Docker, enabling real-time predictions via a simple API.
