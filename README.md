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
 

# Data Exploratory Analysis

## 1) Global analysis of the dataset

## 2) Analysis of important variables

## 3) Remove outliers

## 4) Univariate analysis of quantitative important variables


# Data Modeling

## 1) Standardisation / Normalization
Now, we start to train our data in the differents models which we choose. Firstly, we create a new dataset with the 16 importants variables which we find in our analysis.

 After our analysis, we noticed that it has some outliers and the dataset's distribution was not normal. So, we must normalise it by the RobustScaler method (this method is appropriate when you have many outliers). To use, we must firstly do the log transformation before apply the RobustScaler function. 
 
 Now, our dataset is ready to be trained.
 
 Before training our model, we must divised our dataset with train, test dataset
 
 ## 2) Random Forest Classifier
We first imported our model and then trained it on our data and finally made predictions. ​

We found that our model had a prediction of about 61.27% and a F1_score equal to 64.4%. 

We also pulled the confusion matrix out of the model and find that there are some bad predictions.​



After encoding our classes, we found that :

0 : High ; 1 :Low ; 2 :Medium ; 3 :high 

![image](https://github.com/ChristopherMUFU/python_project/blob/master/images/1.png)

### Change hyperparameters

In order to overcome the bad predictions, we changed the hyper-parameters of our model using the GridSearchCV() function which is a method to find the best possible combination of hyper-parameters at which the model achieved the highest accuracy. The different parameters we have considered here is :​

N_estimators: To control the number of trees inside the classifier​

Max_depth: The depth of the trees which is an important parameter to increase the accuracy of a model

We have obtained 63.58% performance and correctly predicted value and F1_score is 64.1%.

The best values of the hyperparameters are:

max_depth: 16

n_estimators: 256

![image](https://github.com/ChristopherMUFU/python_project/blob/master/images/2.png)

## 3) AdaBoost Classifier

We first imported our model and then trained it on our data and finally made predictions. ​

We found that our model had a prediction of about 59% and F1_score is 66.3%.

We also pulled the confusion matrix out of the model and find that there are some bad predictions.​

After encoding our classes, we found that :

0 : High ; 1 :Low ; 2 :Medium ; 3 :high     

![image](https://github.com/ChristopherMUFU/python_project/blob/master/images/3.png)

### Change hyperparameters

In order to overcome the bad predictions, we changed the hyper-parameters of our model using the GridSearchCV() function which is a method to find the best possible combination of hyper-parameters at which the model achieved the highest accuracy. The different parameters we have considered here is :​

N_estimators: To control the number of trees inside the classifier  

The model obtained the precision of 63.15% and the F1_score of 64%.          

The best values of the hyperparameters are:

N_estimators: 49 

![image](https://github.com/ChristopherMUFU/python_project/blob/master/images/4.png)

## 4) Decision Tree classifier
Same to the previous models, we first imported our model and then trained it on our data and finally made predictions. ​

Our model  has a preformance of 55.42% and the F1_score of 56.2%.  

For the change of hyperparameters, we chose:

Max_depth: to know the best tree depth

Min_sample_split : to know the minimum number of samples required to split an internal code

Min_sample_split : to know minimum number of leaf samples

Criterion : to know the best criterion used to split particular nodes to make decisions

![image](https://github.com/ChristopherMUFU/python_project/blob/master/images/5.png)

### Change hyperparameters

The predictions have obtained 61.26% for performance and 63.1 for the F1_score and the best fitting hyperparameters are : 

criterion: 'entropy',

max_depth: 5, 

min_samples_leaf: 1, 

min_samples_split: 2

![image](https://github.com/ChristopherMUFU/python_project/blob/master/images/6.png)

## 5) Comparaison of model

 To compare the different models, we have ploted the accuracy of this model in the same graph.         ​

       According to the graph, the best model is the random Forest classifier with the highest precision of 63.64.
       
       ![image](https://github.com/ChristopherMUFU/python_project/blob/master/images/7.png)
