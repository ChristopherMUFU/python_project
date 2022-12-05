# Online News Popularity Project
In this repository, we will try to predict the category of an article using differents models of supervised Learning.
The dataset were dowloaded on kaggle website.

## Preparing the datasets
Our goal was to predict if an article is popular or unpopular using machine learning models.

For that, we must prepare the data before using it.

In order to process your data properly, we have made some step.
 ## 1) Extract dataset
 
 First download the dataset from [ here ](https://github.com/ChristopherMUFU/python_project/blob/master/OnlineNewsPopularity.csv) and unpack it.
 
 ## 2) Quick analysis of data
 
 Here, we extract the shape, the types and informations of differents variables of the dataset. We looked also for missing values.
 
 ## 3) Removing rows
 
 We removing rows with null values 
 
 ## 4) Removing columns
 We remove useless columns or columns with highly correlations
 
 ## 5) Transformation of columns
 
 We noticed that some columns contains same informations. We merge theses columns to obtain a cleaner dataset.
 
 ## 6) Study a target variables
 
 To know what we must predict, we have to determinate the target variable of our dataset. Here, our target variable is the shares columns which was an interger column. In order to make it more accurate, we divide it in two category (popular ans unpopular).
 
 II) Data Exploratory Analysis
    1) Global analysis of the dataset
    2) Analysis of important variables
    3) Remove outliers
    4) Univariate analysis of quantitative important variables
    
III) Data Modeling
    1) Standardisation / Normalization
    2) Random Forest Classifier
    3) AdaBoost Classifier
    4) Decision Tree Classifier
# Data Exploratory Analysis

## 1) Global analysis of the dataset

## 2) Analysis of important variables

## 3) Remove outliers

## 4) Univariate analysis of quantitative important variables


# Data Modeling

## Standardisation / Normalization
Now, we start to train our data in the differents models which we choose. Firstly, we create a new dataset with the 16 importants variables which we find in our analysis.

 After our analysis, we noticed that it has some outliers and the dataset's distribution was not normal. So, we must normalise it by the RobustScaler method (this method is appropriate when you have many outliers). To use, we must firstly do the log transformation before apply the RobustScaler function. 
 
 Now, our dataset is ready to be trained.
 
 ## Random Forest Classifier
 
 




