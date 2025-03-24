Project: Implementing Recurrent Neural Networks for Time Series Forecasting and Stock Market Prediction
This project explores the application of Recurrent Neural Networks (RNNs) for time series forecasting and stock market prediction. It consists of two main parts: generating and predicting synthetic time series data, and predicting Google stock prices using historical data.
Part 1: Synthetic Time Series Forecasting
Objective
To generate a synthetic time series dataset and compare the performance of different neural network architectures in predicting the 51st time step.
Code
The synthetic time series dataset is generated using the provided generate_time_series function. The dataset is then split into training, validation, and testing sets in a 70:20:10 ratio. Four different models are trained and evaluated:
Fully Connected Neural Network (FCNN)
Simple RNN with one layer (output layer)
Simple RNN with one hidden layer and one output layer
Simple RNN with two hidden layers and one output layer
Results
FCNN: Achieved a validation loss of 0.0011 and validation MAE of 0.0250 after 30 epochs.
Simple RNN (one layer): Achieved a validation loss of 3.3818e-04 and validation MAE of 0.0149 after 50 epochs.
Simple RNN (one hidden layer): Achieved a validation loss of 2.9333e-05 and validation MAE of 0.0038 after 80 epochs.
Simple RNN (two hidden layers): Achieved a validation loss of 2.0691e-05 and validation MAE of 0.0036 after 80 epochs.
Part 2: Google Stock Price Prediction
Objective
To predict Google stock prices for the years 2020 and 2021 using historical data.
Data Preparation
The dataset is filtered for the years 2020 and 2021. The data is scaled using MinMaxScaler and split into training and testing sets. The input features include scaled close, high, low, open, and volume prices.
Model
A Simple RNN model with 5 layers (including dropout layers) is built and trained for 80 epochs.
Results
MAPE: 0.01%
MSE: 0.0000
Observations
The Simple RNN with two hidden layers performed the best in predicting the synthetic time series data.
For the Google stock price prediction, the model achieved very low MAPE and MSE values, indicating high accuracy.
The number of epochs required for adequate learning varied depending on the complexity of the model and the dataset.
Future Work
Experiment with different RNN architectures (e.g., LSTM, GRU) for better performance.
Incorporate additional features and external data sources to improve prediction accuracy.
Perform hyperparameter tuning to optimize model performance.
Dependencies
Python
TensorFlow
Keras
Pandas
NumPy
Matplotlib
Scikit-learn
