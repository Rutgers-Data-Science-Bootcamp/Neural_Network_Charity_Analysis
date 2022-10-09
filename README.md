# Neural_Network_Charity_Analysis

## Overview of the Project
From Alphabet Soup’s business team, we received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Therefore, using deep-learning neural networks with the TensorFlow platform in Python, to analyze and predict whether applicants will be successful if funded by Alphabet Soup.

## Approaches
We use the following methods for the analysis:
- preprocessing the data for the neural network model,
- compile, train and evaluate the model with test data,
- optimize the model to imporve prediction accuracy.

## Resources
Datasets [Charity Data](https://github.com/ShiraliObul/Neural_Network_Charity_Analysis/blob/main/Resources/charity_data.csv)
Tools: Python, Pandas, Numpy, Scikit, TensorFlow, Jupyter lab

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

### Optimization of model performance
To increase the performance of the model, we applied follwoing steps:
- 1. bucketing to the feature ASK_AMT and organized the different values by intervals.
- 2. increased the number of neurons on one of the hidden layers, then we used a model with three hidden layers.
- 3. use different activation function (tanh) but none of these steps helped improve the model's performance.

## Summary 
The deep learning neural network model did not reach the target of 75% accuracy. Considering that this target level is pretty average we could say that the model is not outperforming.
Since we are in a binary classification situation, we could use a supervised machine learning model such as the Random Forest Classifier to combine a multitude of decision trees to generate a classified output and evaluate its performance against our deep learning model.


