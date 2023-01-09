# Social Media Shares Prediction
![Cool Figure](https://visme.co/blog/wp-content/uploads/2019/07/header-ok.png)


# Team Members

Valerio Romano Cadura

Muhammad Khair Hindawi

Lorenzo Mainetti

---------------------------------------------------------------------------------------------------------

In this project, we were tasked with developing a tool to predict the number of shares on social media given the content and the supposed publication time. To achieve this, used three machine learning models: Neural Network Regression from Keras, XGBoost Regression, and RANSAC Regression. The choice was not casual we first analyse the dataset and understood it was heavily skewed, hence the right path was to choose models capable to adapt to non-linear relationships. We picked these models for the following characteristics:

1. Neural network regression models, such as those implemented in Keras, are good for regression
tasks because they can learn and model non-linear relationships between variables and can handle
large datasets effectively.

2. XGBoost is a fast and accurate model for regression tasks because it is an ensemble model that
combines the predictions of multiple decision trees trained in parallel.

3. RANSAC regression is a method that can fit a model to data with a large amount of outliers or noise by fitting a model to a subset of the data deemed to be inliers and discarding outliers. This makes it suitable for regression tasks when the data is heavily skewed or has a lot of noise. We even considered Support Vector Regression at first glance, but the idea was quickly discarded considering the awful training time given by the 59 features. Another issue that could arise using SVR is that with that many features and relatively few data points the model could have been prone to overfitting.

# Data Pre-processing

Before training the models, we performed an Explanatory Data Analysis (EDA) with visualization to understand the characteristics of the dataset. This included examining the distribution of the target variable (number of shares), as well as the relationship between the target variable and the different features. We also looked for any potential outliers or missing values in the data.

Once we had a good understanding of the data, we generated a training, a cross validation and a test set, with the latter being used only at the end to evaluate the final performance of the model. We used a 52.25-42.75-5 split for the training cross validation and test sets, respectively.
We then pre-processed the data by removing any outliers that we identified during the EDA. Outliers can have a significant impact on the performance of machine learning models, so it is important to remove them before training. We did not have to encode any categorical features using one-hot encoding, as all data was numerical. The validation is meant to analyse the behaviour of the models tuning hyperparameters. This is why we chose to pick a big portion of the dataset as a cross validation set.

# Hyperparameter Tuning
After pre-processing the data, we tuned the hyperparameters for each model using 10-fold cross-validation and Random Search. Initially we were using Grid Search to perform hyperparameter tuning, but we quickly switched to use Random Search. We found out that adding stochasticity to hyperparameter sampling was much faster and still capable to obtain an astonishing sub-optimal solution. Hyperparameter tuning was an important step in the model development process, as the performance of the model was greatly improved finding the sub-optimal set of hyperparameters.

# Model Selection

Once we had tuned the hyperparameters for each model, we selected the best architecture for each model using the appropriate metric. For regression tasks, common evaluation metrics include mean squared error (MSE), mean absolute error (MAE), and R-squared (R2). We selected the model with the lowest (MAE) as the best model, as this metric is not as much sensitive to outliers as (MSE), hence in our case (skewed dataset) in which there are a few extreme outliers, (MAE) is a more robust measure of error.

# Experimental Design

The main purpose of this project is to develop machine learning models that can accurately predict the number of shares on social media given the contents and publication time.

To demonstrate the effectiveness of our approach, we used three different baseline models: Neural Network Regression, XGBoost Regression, and RANSAC Regression. These models were chosen because they are widely used and have been shown to be effective for regression tasks.

We evaluated the performance of the models using three different metrics: mean squared error, R2 score, and mean absolute error. These metrics were chosen because they provide a comprehensive assessment of the model's accuracy and allow us to compare the results of different models.

Additionally, we conducted experiments to demonstrate the impact of feature selection on model performance. We compared the results of the models before and after using two different feature selection methods: PCA and selecting only the columns with a correlation less than alpha (0.05) with the target variable. We also attempted to use variance inflation factor (VIF) to eliminate variables, but this did not result in any improvement in the performance of the models. This allowed us to determine the most effective method for selecting the most relevant features.

# Model Evaluation
After selecting and fitting the best model for each algorithm, we evaluated the performance on the test set.

The performance of the models before and after feature selection is summarized in the following table:
|                     Model                     | Mean Squared Error | R2 Score | Mean Absolute Error |
|:---------------------------------------------:|--------------------|----------|---------------------|
| Neural Network Regression (before)            | 7.09               | -0.15    | 1.12                |
| Neural Network Regression (after PCA)         | 6.83               | -0.11    | 1.1                 |
| Neural Network Regression (after correlation) | 7.362468           | -0.19    | 1.149001            |
| XGBoost Regression (before)                   | 5.75               | 0.07     | 1.29                |
| XGBoost Regression (after PCA)                | 6.1                | 0.01     | 1.37                |
| XGBoost Regression (after correlation)        | 5.72484            | 0.072464 | 1.290114            |
| RANSAC Regression (before)                    | 7.42               | -0.2     | 1.17                |
| RANSAC Regression (after PCA)                 | 7.1                | -0.15    | 1.12                |
| RANSAC Regression (after correlation)         | 7.32               | -0.19    | 1.16                |

The scores for each model are as follows:

Neural Network Regression:

o Mean Squared Error: 6.48
o R2 Score: -0.05
o Mean Absolute Error: 1.09

XGBoost Regression:

o Mean Squared Error: 5.75
o R2 Score: 0.07
o Mean Absolute Error: 1.29

RANSAC Regression:

o Mean Squared Error: 7.4
o R2 Score: -0.2
o Mean Absolute Error: 1.17

As we can see, (MSE) and (MAE) are low but in NN and RANSAC Regression the model is not capable of explaining any of the variance in the data. Regarding XGBoost the model is fitting the data, the measure is low but it can still be considered a starting point right now is the only model capable of outperforming a model that always predict the mean of the data (benchmark).

# Feature Selection

To further improve the performance of the best model, we selected a subset of the attributes retaining only the most relevant features. There are many ways to select the most important features in a dataset our first pick was to use variance inflation factor to determine multicollinear columns and drop them, this first approach was a disaster the model was not fitting data at all, and we obtained pretty insane loss
function. In the end we opted to choose two different methods: Principal Component Analysis (PCA) and P-Value Statistical Significance.

PCA was intended to solve the same issue VIF could: remove multicollinearity, and in some cases performing PCA can lead to improved model performance by removing noise and irrelevant features from the dataset. Sadly it was not our case.

P-Value Statistical Significance aimed to drop statistically insignificant variables. This is considered a good practice in a regression task as it can reduce the risk of overfitting by reducing the number of features in the mode. Features with a P-Value greater than 0.05 indicates that the feature is not statistically significant and is not likely to have an effect on the outcome.

*Principal Component Analysis (PCA)*

PCA is a dimensionality reduction technique that projects the data onto a lower-dimensional space, retaining the most important features of the data. This can be useful for reducing the number of features in a dataset, as it allows us to select the most important features without having to manually identify and remove the less important ones. To use PCA for feature selection, we first fit the PCA model to the training data and then transform the data using the model. We can then select the number of components to retain based on the explained variance of each component. In this project, we retained the top 2 components that explained the 95% of variance in the dataset.

After using PCA, we obtained the following results: 

*Neural Network Regression*:

o Mean Squared Error: 6.88
o R2 Score: -0.11
o Mean Absolute Error: 1.1

*XGBoost Regression*:

o Mean Squared Error: 6.1
o R2 Score: 0.01
o Mean Absolute Error: 1.29

*RANSAC Regression*:

o Mean Squared Error: 7.12
o R2 Score: -0.15
o Mean Absolute Error: 1.12

As we can see, the performance of the RANSAC Regression model improved slightly, still no major improvements. We can say that PCA was not a good choice.

*P-Value Statistical Significance*

Another method for selecting the most important features is to use the correlation between the features and the target variable. Features with a high correlation with the target variable are likely to be more important for predicting the target. To select features using correlation, we first fit a linear regression model to the data and compute the p-values for each feature. Features with a p-value greater than a certain threshold (in this case, 0.05) are considered to be not significant and are removed from the dataset. This is known as feature selection through p-value thresholding. This process can be useful for reducing the risk of overfitting by removing features that are not likely to have a meaningful impact on the outcome. After dropping variables using this method, we obtained the following results:

*Neural Network Regression*:

o Mean Squared Error: 1.57
o R2 Score: 0.74
o Mean Absolute Error: 0.32

*XGBoost Regression*:

o Mean Squared Error: 0.00024
o R2 Score: 0.999967
o Mean Absolute Error: 0.002864

*RANSAC Regression*:

o Mean Squared Error: 0.6
o R2 Score: 0.9
o Mean Absolute Error: 0.18

As we can see, the performance of all three models significantly improved after using P-Value for feature selection. The XGBoost Regression model in particular achieved a considerable score, with an MSE of 0.00024 and an R2 score of 0.999967.

# Final Thoughts and Conclusions

Based on these results, the best model for predicting the number of shares on this dataset is the XGBoost Regression model, with the best performance being achieved after using feature selection through P-Value. It is important to notice that in some cases, RANSAC regression may be a better choice than XGBoost, especially if the data is heavily skewed or has a large amount of noise. This is because RANSAC is able to fit a model to the inlier data while ignoring the outlier data, which can lead to improved model performance. In our case XGBoost was the model that since the first trials had most success applied to data provided. However, it is important to note that the results of this study are specific to the dataset and the models used and may not necessarily generalize to other datasets or models. Further research is needed to confirm the effectiveness of this tool for predicting the number of shares on social media.

# Further Recommendations:

There are several ways in which the performance of the models could be further improved. First, we could try using more advanced models, such as GANs or other type of ensembles of multiple models. These models have the potential to achieve better performance, but they may also be more
complex to implement and require more computational resources. Predictive power of regressor variables could be assessed with PPScore, it can be useful in the analysis of skewed datasets because it takes into account the class imbalance in the data and adjust the score accordingly. (We figured out this package exists too late in order to implement in the project). 
Another option is to try different feature engineering techniques to extract more meaningful features from the data. This could include techniques such as feature selection through mutual information or creating new features through combination or transformation of existing features.

In addition, we could try using different evaluation metrics to select the best model. While MSE is a commonly used metric for regression tasks, other metrics such as Mean Absolute Percentage Error (MAPE) or Quantile Loss may be more suitable for certain types of data.

In conclusion, there are many ways in which the performance of the models can be further improved.
However, it is important to carefully consider the trade-offs between performance and complexity when
selecting the appropriate methods and models. Further research is needed to identify the most effective
approaches for predicting the number of shares on social media.
