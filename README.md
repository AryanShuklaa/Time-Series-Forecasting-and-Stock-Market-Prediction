# 📈 Implementing Recurrent Neural Networks for Time Series Forecasting & Stock Market Prediction  

This project explores the application of **Recurrent Neural Networks (RNNs)** for **time series forecasting** and **stock market prediction**. It consists of two main parts:  
1️⃣ **Generating and predicting synthetic time series data**  
2️⃣ **Predicting Google stock prices using historical data**  

---

## **🔹 Part 1: Synthetic Time Series Forecasting**  

### **🎯 Objective**  
To generate a synthetic time series dataset and compare the performance of different **neural network architectures** in predicting the **51st time step**.  

### **📌 Methodology**  
- The **synthetic time series dataset** is generated using the `generate_time_series` function.  
- The dataset is split into **training (70%), validation (20%), and testing (10%)**.  
- **Four models** are trained and evaluated:  
  1️⃣ **Fully Connected Neural Network (FCNN)**  
  2️⃣ **Simple RNN (1 layer - output only)**  
  3️⃣ **Simple RNN (1 hidden layer + output layer)**  
  4️⃣ **Simple RNN (2 hidden layers + output layer)**  

### **📊 Results**  

| Model | Validation Loss | Validation MAE | Epochs |
|--------|----------------|----------------|--------|
| **FCNN** | 0.0011 | 0.0250 | 30 |
| **Simple RNN (1 layer)** | 3.3818e-04 | 0.0149 | 50 |
| **Simple RNN (1 hidden layer)** | 2.9333e-05 | 0.0038 | 80 |
| **Simple RNN (2 hidden layers)** | **2.0691e-05** | **0.0036** | 80 |

💡 **Observations:**  
✔ The **Simple RNN with two hidden layers** performed best, achieving the lowest validation loss and MAE.  

---

## **🔹 Part 2: Google Stock Price Prediction**  

### **🎯 Objective**  
To predict **Google stock prices** for the years **2020 and 2021** using historical data.  

### **📌 Data Preparation**  
- The dataset is filtered for **2020 and 2021**.  
- The data is **scaled** using `MinMaxScaler` and split into **training and testing sets**.  
- Input features include **Close, High, Low, Open, and Volume prices (scaled)**.  

### **🛠️ Model Architecture**  
A **Simple RNN model** with **5 layers (including dropout layers)** is built and trained for **80 epochs**.  

### **📊 Results**  

| Metric | Value |
|--------|---------|
| **MAPE (Mean Absolute Percentage Error)** | 0.01% |
| **MSE (Mean Squared Error)** | 0.0000 |

💡 **Observations:**  
✔ The **Simple RNN with two hidden layers** was the best performer for **synthetic time series data**.  
✔ For **Google stock price prediction**, the model achieved **very low MAPE and MSE values**, indicating **high accuracy**.  
✔ The **number of epochs required** varied depending on the **model complexity** and **dataset size**.  

---

## **🚀 Future Work**  
🔹 Experiment with different RNN architectures (**LSTM, GRU**) for better performance.  
🔹 Incorporate **additional features** (e.g., macroeconomic indicators, news sentiment) to improve accuracy.  
🔹 Perform **hyperparameter tuning** to optimize the model.  

---

## **📦 Dependencies**  

✔ Python  
✔ TensorFlow  
✔ Keras  
✔ Pandas  
✔ NumPy  
✔ Matplotlib  
✔ Scikit-learn  

📌 **To install all dependencies, run:**  
```bash
pip install tensorflow keras pandas numpy matplotlib scikit-learn
