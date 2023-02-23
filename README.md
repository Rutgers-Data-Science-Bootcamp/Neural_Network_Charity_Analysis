# Neural_Network_Charity_Analysis

## Overview of the Project
From Alphabet Soup’s business team, we received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Therefore, using deep-learning neural networks with the TensorFlow platform in Python, to analyze and predict whether applicants will be successful if funded by Alphabet Soup.

## Approaches
We use the following methods for the analysis:
- preprocessing the data for the neural network model,
- compile, train and evaluate the model with test data,
- optimize the model to imporve prediction accuracy.

## Resources
- Dataset: [Charity Data](https://github.com/ShiraliObul/Neural_Network_Charity_Analysis/blob/main/Resources/charity_data.csv)
- Tools: Python, Pandas, Numpy, Scikit, TensorFlow, Jupyter lab

## Results 
### Processing the dataset in order to compile, train, and evaluate the neural network model
There are a number of columns that capture metadata about each organization, such as the following:
- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special consideration for application
- ASK_AMT—Funding amount requested
- IS_SUCCESSFUL—Was the money used effectively

Processing steps:
- Drop the columns EIN and NAME  that are identification information which have no value in modeling.
- The column IS_SUCCESSFUL is the target feature for our deep learning neural network as it contains binary data refering to weither or not the charity donation was used effectively.
- These columns APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT are the features for our model.
- Encoding of the categorical variables, spliting into training and testing datasets and standardization have been applied to the features.

Original dataset were 12 columns and 34299 rows:
![Screen Shot 2022-10-09 at 6 33 37 PM](https://user-images.githubusercontent.com/65901034/194782524-1e631a8b-025e-456b-ae24-bc7a5d71d7a7.png)

After processing for Neural network model, there is 44 columns and 34299 rows: 
![Screen Shot 2022-10-09 at 6 40 56 PM](https://user-images.githubusercontent.com/65901034/194782831-82ef8e9f-b08c-46cc-b5da-8a75abaf2794.png)
### Compiling, Training, and Evaluating the Model 
- As we processed the data, the input data has 43 features and 34299 samples.
- We did random split the data with train dataset and test dataset, then scale the data to make them ready for model application.
- We set a deep-learning neural network model with two hidden layers with 8 and 5 neurons respectively.
- We set a output layer with a unique neuron as it is a binary classification.
- To speed up the training process, we are using the activation function ReLU for the hidden layers. As our output is a binary classification, Sigmoid is used on the output layer.
- For the compilation, the optimizer is adam and the loss function is binary_crossentropy.

Here is the model summary:

![Screen Shot 2022-10-09 at 6 48 37 PM](https://user-images.githubusercontent.com/65901034/194783073-41728c19-d568-4ab6-8d27-041efbc2690d.png)

- As you see in the screenshot, with both train and test data, we reached the accuracy about 73%. This is not a satisfying performance to help predict the outcome of the charity donations as try to reach more than 75%

![Screen Shot 2022-10-09 at 6 47 52 PM](https://user-images.githubusercontent.com/65901034/194783055-63d96f7d-94cb-4264-9220-07dc71adc179.png)

**[`^        back to top        ^`](#Overview-of-the-Project)**

### Optimization of model performance
To increase the performance of the model, we applied follwoing steps:
- 1. bucketing to the feature ASK_AMT and organized the different values by intervals.
- 2. increased the number of neurons on one of the hidden layers, then we used a model with three hidden layers.
- 3. use different activation function (tanh) but none of these steps helped improve the model's performance.

The optimization results can be seen here: [results](https://github.com/ShiraliObul/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity_Optimzation.ipynb)

### Save the model 
we used the Keras Sequential model's save method to export the entire model (weights, structure, and configuration settings) to an Hierarchical Data Format file. The model can be downloaded here [Model saved file](https://github.com/ShiraliObul/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.h5). With thi H5 file, one can import the exact same trained model to their environment by using the Keras load_model method and use it for analysis.

## Summary 
Even with the 3 different optimization attempts, our deep learning neural network model did not reach 75% accuracy with longer time consuming to process. Then we tried supervised logistic regression machine learning model, Random Forest Classifier to combine a multitude of decision trees to generate a classified output and evaluate its performance against our deep learning model. As you can see in the screenshot below, the result was instantaneous as compared to deep learning model, moreover, the accuracy was about 73%. 
![Screenshot 2023-02-23 at 7 40 52 AM](https://user-images.githubusercontent.com/65901034/220909562-94e84688-ce8a-46a0-8467-8e61bc1dccac.png)

The conclusion is, there is no garantee that deep learning models can always outperform machine learning. Beyond the performance of both models, there are a few other factors to consider when selecting a model for our data. First, neural networks are prone to overfitting and can be more difficult to train than a straightforward logistic regression model. Therefore, if we are trying to build a classifier with limited data points (typically fewer than a thousand data points), or if our dataset has only a few features, neural networks may be overcomplicated. Additionally, logistic regression models are easier to dissect and interpret than their neural network counterparts, which tends to put more traditional data scientists and non-data experts at ease. In contrast, neural networks (and especially deep neural networks) thrive in large datasets. Datasets with thousands of data points, or datasets with complex features, may overwhelm the logistic regression model, while a deep learning model can evaluate every interaction within and across neurons. Therefore, the decision between using a logistic regression model and basic neural network model is nuanced and, in most cases, a matter of preference for the data scientist.
**[`^        back to top        ^`](#Overview-of-the-Project)**


