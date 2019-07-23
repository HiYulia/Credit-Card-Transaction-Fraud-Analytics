# Credit-Card-Transaction-Fraud-Analytics
## Summary
We built a supervised model to identify fraudulent events in the credit card transaction data on companies in Tenness during the year of 2010.

Firstly, We did data exploration, data cleaning, candidate variables, and feature selection. 
Then, We used five algorithms(Logistic Regression, Naïve Bayes, Random Forest, Boosted Trees, and Neural Nets) yo build models. 
In the end, I presented the results and conclusions and insights.

## Data Cleaning

Before filling missing values, we made three adjustments to the dataset:

• Only included the values with [Transtype] P, or present
• Excluded the outlier record with the maximum transaction amount dramatically higher than the rest of records
• Changed all invalid state abbreviations to ‘others’

The fields that required cleaning and filling missing values include [Merch state], [Merch zip] and [Merchnum]. 

The general idea of filling missing fields is to use summary statistics of similar groups that the records identify with while minimizing the likelihood of causing any unwanted abnormal results. 

## Candidate Variables
After cleaning the data, we then created variables for further analysis. There are four types of variables, and the steps of variable creation are listed as the following:

a. Amount variables

First, we grouped all the records respectively by Cardnum, Merchnum, [Cardnum & Merchnum], [Cardnum & Merch zip], and [Card & Merch state].

For each group g, we calculated the average, maximum, median, and the total amount over the past 0, 1, 3, 7, 14, and 30 days respectively.

Based on the results above, for each group g, we then calculated the [actual/average], [actual/maximum], [actual/median], [actual/total] amount over the past 0, 1, 3, 7, 14, and 30 days respectively.

In this step, we created 5 x 8 x 6 = 240 amount variables.

b. Frequency variables

Same as above, we grouped all the records respectively by Cardnum, Merchnum, [Cardnum & Merchnum], [Cardnum & Merch zip], [Card & Merch state].

For each group g, we counted the number of transactions over the past 0, 1, 3, 7, 14, and 30 days respectively.

In this step, we created 5 x 6 = 30 frequency variables.

c. Days since variables

Same as above

d. Velocity change variables

To create the velocity change variables, we first grouped all the records respectively by Cardnum, and Merchnum.
For each group g, we then calculated the [number] and [amount] of transactions over the past 0 and 1 day respectively.
For each group g, we also calculated the average daily [number] and [amount] of transactions over the past 7,14, and 30 days respectively.
Then, we calculated the velocity change variables using the formula below:

In this step, we created 2 x 2 x 2 x 2 x 2 x 3 = 96 velocity change variables.

Overall, we created 240 + 30 + 5 + 96 = 371 variables in total.

## Feature Selection Process

Before moving on to the feature selection process, we separated the data into training, testing, and out-of-time (OOT) validation data sets to ensure the proper treatment of time. 

Next, with the newly created 371 variables, we performed feature selection on the training and testing dataset using filter, wrapper, and embedded methods. During the univariate filter step, we removed about 2/3 of the variables, leaving 123 variables. Then, we reduced the number of variables to 20 using the wrapper method, with a stepwise logistic regression. On the final dataset, we used regularization while exploring a handful of nonlinear models. More details are given below.

### Filter
We calculated the KS and Fraud Detection Rate at 3% for each variable. Then we ranked them in descending order and selected the top 123 for further process.

### Wrapper
we chose 123 variables to perform our next step of feature selection  In this step, we used the backward stepwise selection to screen out 20 variables as the input of our models, which are listed as the table below.

### Embedded
The last step in the feature selection is the embedded method, which includes 1) decision trees and 2) regularization.
Since using all features to build a decision tree model is very likely to overfit the testing data, we usually select a certain group of variables with the highest explanatory power. In this project, though we did not directly use the decision tree, we did use the random forest, which took effect in the final model performance.

On the other hand, regularization is adding a penalty term to the loss function while building a very complex model or a model with too many parameters. In the case of linear regression, this is interpreted as Ridge or Lasso regression. In the case of tree algorithms, the penalty term might be a parameter multiplied by the number of nodes in the tree to reduce the complexity of the model. We used regularization in some of the models we tried.


## Model Algorithms


