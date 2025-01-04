# Credit Card Payment Delinquency Prediction Model

This project develops a machine learning model to predict credit card payment delinquency, emphasizing transparency, data-driven feature selection, and an exploration of various models. It also aims at providing an alternative to opaque scoring models.

## Project Overview

The primary goal of this project is to build a predictive model that can effectively classify individuals based on their likelihood of being late on credit card payments. The project prioritizes a transparent approach and aims to understand the key factors driving payment delinquency.

## How This Project Came About

This project arose from the need to explore how machine learning could offer a more transparent, fair, and effective alternative to existing credit scoring models. We were inspired by the challenge of creating models that not only predict risk but also offer clear, interpretable insights into the factors driving those predictions, and address the needs of those who do not have access to credit through the traditional credit channels. This was undertaken as an academic project aimed at making a difference in the financial community by providing insight into financial inclusivity.

## Motivation

Our motivation stems from a commitment to addressing the 'black box' nature of many traditional credit models. We were further motivated by the observation that millions of individuals lack access to traditional credit services. By developing a transparent model, we hope to contribute to greater financial inclusivity, enabling these individuals to participate in the financial system, and enhance the process of making informed decisions based on understandable metrics.

## Approach

This project uses a variety of data manipulation, feature extraction, and machine learning techniques to reach its goals:
*  **Data Acquisition:** We use two datasets from Kaggle, which contain customer information, credit history, and financial data.
*  **Data Cleaning and Preprocessing:** Steps include removing a column with 33% missing values from first dataset due to low correlation with the target, dropping rows with missing values from our merged data set, identifying and removing outliers in the AMT_INCOME_TOTAL column, and data normalization.
*   **Feature Engineering:** We implemented aggregation techniques for the credit records dataset, created custom bins for the `MONTHS_BALANCE_mean` variable, conducted PCA for a set of dummy variables with high correlation, and transformed categorical variables into numerical format.
*   **Feature Selection:** We select features based on domain knowledge, Random Forest analysis, and feature correlation.
*   **Data Partitioning:** The dataset is split into 70/30 for training and testing. We used oversampling for dealing with an unbalanced dataset.
*   **Model Development**: We utilize various models such as Logistic Regression, Decision Tree, XGBoost, Random Forest, Gradient Boosting and SVM to establish a benchmark for our models.
*   **Model Training and Evaluation**: We trained our model on training data, and evaluated it on the test data using a range of techniques for optimization.

## Dataset Overview

*   Data Source: The data was obtained from Kaggle.
*   Datasets: [kaggle](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)
    *   `application_record.csv`: Contains application data with demographics, financial information, and other details.
    *   `credit_record.csv`: Contains records of past credit history.
*   The dataset included a large number of applicants and credit history records, with a wide variety of features.
* The data was analyzed to identify key features for predicting payment delinquency.

## Key Features

*   **Transparent Modeling:** A focus on using methods that offer clear insights into predictions, avoiding the black box problem.
*   **Feature Engineering:** Creation of new, more informative features from the existing data.
*   **Feature Selection:** Analysis and selection of relevant features using statistical tests, model-based selection, or iterative methods.
*   **Model Performance**: Evaluation of various machine learning models including a comprehensive comparison.
* **Actionable Insights:** Emphasis on identifying key factors (payment history) which drive the prediction of credit delinquency.

## Files

*   `src/`: Contains python source files used for building the model, including feature engineering and model comparison.
    *   `python_project-3.py` : The file with the final steps for building and evaluating model.
    *  `pythonproject__dataexploration&preproccessing-9.py`: The data exploration and pre-processing steps which helps generate the data frame.
*   Presentation file: The presentation contains a summary of findings, analysis and insights, and next steps.
* Raw dataset: [kaggle](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)

## Technologies Used

*   Python (for data analysis and model implementation)
*   Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels, imbalanced-learn, xgboost

## Intended Use

This model is designed to be used to:
-  Enhance transparency for prediction of credit payments.
-   Provide transparent insights into key factors driving credit payment delinquency, which can be used for making more informed decisions.
-   Help identify those in need of credit through targeted outreach and tailored products.

## Limitations

*   **Data Limitations:** The data may not fully capture the nuances of individual financial situations. The models are constrained by the available features and variables.
*   **Label Limitations:** The model relies on the derived 'Target Variable', which is a proxy for late payments. This is not necessarily a perfect representation of default risk.
*   **Generalization:** Performance of the model may vary when applied to new datasets with different characteristics.

## Challenges

*   **Defining the Target Variable:** Creating a suitable proxy label for creditworthiness using available information, to address the lack of a predefined target.
*   **Handling Unbalanced Data:** Addressing the issue of imbalanced classes in our target variable, with far fewer instances of delinquency than non-delinquency.
*   **Interpreting Model Results:** Balancing accuracy with transparency, ensuring the model's predictions are not only correct, but also easy to explain and understand.

## How to Use

1.  Clone this repository.
2.  Navigate to the `src` directory and use either of the python source files for analysis and reproduction of models.
3. Ensure all libraries are installed.

## Notes

Missing Files: Due to the nature of an academic project, please note that some of data cleaning and imputation files may not be included in this repository.
