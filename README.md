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

![Image of velocity](https://github.com/xinyueniu/Credit-Card-Transaction-Fraud-Analytics/blob/master/Velocity.png)

In this step, we created 2 x 2 x 2 x 2 x 2 x 3 = 96 velocity change variables.

Overall, we created 240 + 30 + 5 + 96 = 371 variables in total.

## Feature Selection Process

Before moving on to the feature selection process, we separated the data into training, testing, and out-of-time (OOT) validation data sets to ensure the proper treatment of time. 

Next, with the newly created 371 variables, we performed feature selection on the training and testing dataset using filter, wrapper, and embedded methods. During the univariate filter step, we removed about 2/3 of the variables, leaving 123 variables. Then, we reduced the number of variables to 20 using the wrapper method, with a stepwise logistic regression. On the final dataset, we used regularization while exploring a handful of nonlinear models. More details are given below.

### Filter
We calculated the KS and Fraud Detection Rate at 3% for each variable. Then we ranked them in descending order and selected the top 123 for further process.

For Classification: The more separate the distributions (good and bad) are, the better this feature is.

	Kolmogorov-Smirnov (KS)
	Fisher Score
	Pearson Correlation
	Kulback-Leibler(KL=∫▒〖d_s p_1 log p_1/p_2 〗-∫▒〖d_s p_2 log p_1/p_2 〗=∫▒〖d_s (p_1-p_1)log p_1/p_2 〗)
	Information Value
	Mutual Information
  
KS you want it to be high. You rank it. 


### Wrapper
we chose 123 variables to perform our next step of feature selection  In this step, we used the backward stepwise selection to screen out 20 variables as the input of our models, which are listed as the table below.

•	Backward selection: Build a single model using all variables. Next build n models each removing one variable. Select the best model; it has n-1 variables. Then build n-1 separate models each removing one variable. Select the best model. Continue until the model degradation is below an acceptable amount.
recursive feature elimination python (RFE)


### Embedded
The last step in the feature selection is the embedded method, which includes 1) decision trees and 2) regularization.
Since using all features to build a decision tree model is very likely to overfit the testing data, we usually select a certain group of variables with the highest explanatory power. In this project, though we did not directly use the decision tree, we did use the random forest, which took effect in the final model performance.

On the other hand, regularization is adding a penalty term to the loss function while building a very complex model or a model with too many parameters. In the case of linear regression, this is interpreted as Ridge or Lasso regression. In the case of tree algorithms, the penalty term might be a parameter multiplied by the number of nodes in the tree to reduce the complexity of the model. We used regularization in some of the models we tried.


## Model Algorithms

### Model 1: Logistic Regression
Logistic Regression can model a binary dependent variable. When doing Logistic Regression, we want to know that given X = [x1, x2, ..., xn]T, what is the probability of Y=1 happens. This method can be used in situations like predicting the probability of a sunny or cloudy weather tomorrow, given today’s weather and today’s temperature. Here, Xs are the weather and the temperature today, Y is the weather tomorrow, and 1 means sunny while 0 means not sunny (or cloudy). And the way of calculating such probability is by using the formula below.

![Image of lr](https://github.com/xinyueniu/Credit-Card-Transaction-Fraud-Analytics/blob/master/Logistic%20Regression.png)

In the formula above, x is a vector with n rows and one column, b is the coefficient vector with n rows and one column, x prim means the transposition of vector x. To estimate b in the Logistic Regression, we use Maximum Likelihood Estimation. After estimating b, when we have x and want to know the likelihood of Y=1 happening, we can plunge x in the formula above and get the probability of Y=1.

Parameters:
- C=1/lambda: Inverse of regularization strength; must be a positive float. Smaller values specify stronger regularization.
- random_state: The seed of the pseudo random number generator to use when shuffling the data.

### Model 2: Naïve Bayes
Naïve Bayes is a classifier of machine learning based on Bayes’ theorem assuming naive independence between all variables. Unlike other machine learning models that try to predict Y given X, Naïve Bayes predicts, given Y, how likely the records display features of X. After building the model, we would be able to use Bayes theorem to calculate, given new X, the probability of Y being any class and choose the most likely predicted result. The following steps show how the model works.

![Image of Naïve Bayes](https://github.com/xinyueniu/Credit-Card-Transaction-Fraud-Analytics/blob/master/Nai%CC%88ve%20Bayes.png)

### Model 3: Random Forest
Random Forest is a bagging technique for both classification and regression based on a decision tree. It solves Decision Tree’s problem of finding the right tree depth, as it reduces the variance by averaging multiple deep decision trees trained on different parts of the same training set. This comes at the expense of a small increase in the bias and some loss of interpretability, but Random Forest greatly boosts the performance in the final model generally.

Parameters:

- n_estimators: The number of trees in the forest. The more estimators usually mean a
better performance. 500 or 1000 is usually sufficient.
- max_features: The number of features to consider when looking for the best split.
- max_depth: The maximum depth of the tree. Reduction of the maximum depth helps
fight with overfitting. If nothing is given to this parameter, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

### Model 4: Boosted Trees
Boosted Decision Tree is a machine learning algorithm that produces a prediction model in the form of an ensemble of weak classifiers which are decision trees in this case. Given dataset (X(1) ,y(1)), ... ,(X(n), y(n)):

• Initially assign every point equal weight;
• Repeatfort=1,2,...:
• Feed weighted dataset to the decision tree and get a weak classifier dt
• Reweight the data to put more emphasis on points that dt gets wrong
• Combine all the dt linearly

Parameters:

- learning rate: shrinks the contribution of each tree
- number of estimators: the number of boosting stages to perform
- criterion: the function to measure quality of a split

### Model 5: Neural Nets

Neural Net is a mathematical function mapping inputs to an output with a set of adjustable parameters. A typical neural net consists of an input layer, number of hidden layers and an output layer. An input layer has all the independent variables. An output layer refers to the dependent variable. The hidden layer contains a set of nodes. Each node in each hidden layer contains a linear combination of all the nodes in the previous layer and does a transform on this linear combination. The transform function can be a logistic function, a step function, a linear function, etc.

Parameters:

- number of inputs: independent variables in the dataset
- number of hidden layers: it depends on different situations
- number of nodes in each hidden layer: it depends on different situations
- transform function: a logistic function, a step function, a linear function, etc.
- learning rate: a hyper-parameter that controls how much we are adjusting the weights of our model with respect the loss gradient.

In total, we tried five neural networks and it turned out that the model with three hidden layers and 128 neurons within each had the best performance. The first two hidden layers used ‘tanh’ as activation function and the third used ‘relu’.

### performance

![Image of performance](https://github.com/xinyueniu/Credit-Card-Transaction-Fraud-Analytics/blob/master/performance.png)

## Results
According to the above table demonstrating a high-level overview of all five models’ FDRs at 3% in training, testing and out-of-time dataset, both random forest and boosted tree performed well on the training set; boosted tree did better in the testing set while random forest did better in out-of-time set. In the end, we chose the boosted tree as our final model. Using top 3% of the population with highest predictions, the boosted tree model achieved a 100% fraud detection rate on the training set, 88% on testing, and 37% on out-of-time dataset, respectively.
We also generated three tables that showcase the final model performance in training, testing and out of time datasets separately. In each of them, we collected their bin statistics and cumulative statistics according to the population bin.


## Fraud Savings Calculation:

We created the graph below to show our fraud algorithm savings. We assumed that we could gain $2000 for every true fraud we caught (blue curve) and lose $50 for every inaccurately identified fraud (red curve). Then, the overall savings (grey curve) is equal to fraud savings minus lost sales. Since we would like to save as much money as possible, as demonstrated below, the overall saving reached the highest point of $140,550 when targeting the top 14% of population with highest predictions. Therefore, we recommend that the client set a cutoff point at 14%.

![Image of fsc](https://github.com/xinyueniu/Credit-Card-Transaction-Fraud-Analytics/blob/master/Fraud%20Savings%20Calculation.png)
## cros validation3
## out of time and test side
