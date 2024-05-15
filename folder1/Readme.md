# Assessment ReadMe File

**Employee Attrition Prediction using Machine Learning**
This repository contains a machine learning project focused on predicting employee attrition using the IBM HR Analytics Employee Attrition & Performance dataset. The project involves data preprocessing, model development, evaluation, and optimization.

**Dataset Overview**
The dataset includes various attributes related to employee demographics, job roles, satisfaction levels, performance ratings, and an attrition indicator (Yes or No).

**Process**
Data Preprocessing

Handled missing values using forward fill (ffill) method.
Encoded categorical variables using OneHotEncoder.
Split the dataset into training (80%) and testing (20%) sets.
Model Development and Evaluation

Utilized two classifiers:

Random Forest Classifier
Achieved an accuracy of 0.8776%, precision of 0.8000 , Recall: 0.1026, F1-score: 0.1818.

Support Vector Classifier (SVC)
Attained an Accuracy: 0.8810, Precision: 1.0000, Recall: 0.1026, F1-score: 0.1860.

Model Optimization

Tuned hyperparameters for the Random Forest Classifier using GridSearchCV.

Insights and Recommendations
Key Findings

Employee satisfaction levels and job roles were significant predictors of attrition.
SVC outperformed Random Forest Classifier in terms of accuracy and F1-score.
Challenges Encountered

Addressing class imbalance.
Efficient hyperparameter tuning.
Recommendations for Reducing Employee Attrition

Conduct regular employee satisfaction surveys.
Implement personalized retention strategies.
Foster a positive work environment and provide professional development opportunities.

**Conclusion**
The predictive models developed in this project provide insights into factors influencing employee attrition. By leveraging machine learning techniques and optimization methods, organizations can proactively mitigate attrition risks and enhance employee retention strategies.

This README provides a summary of the project, including its objectives, methodology, key findings, and recommendations.
