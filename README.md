# **S&P 500 Stock Market Prediction**

This repository contains two projects that aim to predict the S&P 500 stock prices using machine learning models. The projects utilize historical data spanning five years (2019-2024) and apply both **Linear Regression** and **Long Short-Term Memory (LSTM)** neural networks for stock market prediction.

## **Table of Contents**

1. [S&P 500 Stock Market Prediction - Using Linear Regression](#s&p-500-stock-market-prediction---using-linear-regression)
   - [Project Overview (LR)](#project-overview-lr)
   - [Dataset Information (LR)](#dataset-information-lr)
   - [Exploratory Data Analysis (EDA) (LR)](#exploratory-data-analysis-eda-lr)
   - [Model Development (LR)](#model-development-lr)
   - [Results (LR)](#results-lr)
   - [Financial Analysis (LR)](#financial-analysis-lr)
   - [Conclusion (LR)](#conclusion-lr)
   
2. [S&P 500 Stock Market Prediction - Using Long Short-Term Memory (LSTM)](#s&p-500-stock-market-prediction---using-long-short-term-memory-lstm)
   - [Project Overview (LSTM)](#project-overview-lstm)
   - [Dataset Information (LSTM)](#dataset-information-lstm)
   - [Model Architecture (LSTM)](#model-architecture-lstm)
   - [Training Process (LSTM)](#training-process-lstm)
   - [Model Evaluation Metrics (LSTM)](#model-evaluation-metrics-lstm)
   - [Detailed Analysis of the Metrics (LSTM)](#detailed-analysis-of-the-metrics-lstm)
   - [Predicted vs Actual Stock Prices Financial Implications (LSTM)](#predicted-vs-actual-stock-prices-financial-implications-lstm)
   - [Conclusion (LSTM)](#conclusion-lstm)

---

# **S&P 500 Stock Market Prediction - Using Linear Regression**

## **Introduction**
This project aims to analyze and predict the stock prices of the S&P 500 index using historical data spanning over five years. The dataset includes various attributes such as **Open**, **High**, **Low**, **Close** prices, and **Volume**, which are utilized to build a linear regression model. The goal is to predict the closing prices based on the historical trends of the other variables.

---

## **Project Overview (LR)**

This project leverages machine learning to predict the closing price of S&P 500 stocks using historical data from August 11, 2019, to August 9, 2024. A **Linear Regression** model is developed to predict future stock prices based on features like Open, High, Low prices, and Volume. The model is evaluated using metrics such as:

- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **R² score**

These metrics assess the model's performance and predictability.

---

## **Dataset Information (LR)**
The dataset comprises stock price data from 2019 to 2024. It is loaded into a **Pandas DataFrame** with the following key columns:
- **Date**: The timestamp of the stock prices (renamed from "Time" to "Date" for clarity).
- **Open**: Opening price of the S&P 500 index at the given time.
- **High**: The highest price recorded during the given time frame.
- **Low**: The lowest price recorded during the given time frame.
- **Close**: The closing price of the S&P 500 index.
- **Volume**: The total volume of stocks traded.

---

## **Exploratory Data Analysis (EDA)**

### **1. Descriptive Statistics**
By using the `describe()` function, we obtained summary statistics that provide insight into the range, central tendency, and variability of the stock prices. Below are the key statistics for the stock’s **Open**, **High**, **Low**, **Close** prices, and **Volume**:

#### **Interpretation:**
- The mean **Open** and **Close** prices of around **4327** show that the stock market was relatively stable over the five-year period, though there were some significant fluctuations.
- The minimum value of **2463.50**, recorded in 2019, likely corresponds to a market dip, possibly associated with external factors like economic downturns or global events (e.g., the **COVID-19 pandemic**).
- The maximum **Close** price of **5719.00** shows how much the index has grown over this time frame, reflecting an upward trend in the S&P 500.
- The volume of trades varies widely, from very low numbers (1 trade) to extremely high numbers (**262,213** trades), indicating periods of both low and high liquidity in the market.
![image](https://github.com/user-attachments/assets/9133d07e-00a7-49a3-8b7d-356a210b44bc)


### **2. Box Plot Analysis**
We plotted box plots for the **Open**, **High**, **Low**, and **Close** prices to visualize the spread and identify outliers in the data. This analysis helps detect unusual data points that could potentially skew model predictions.
![image](https://github.com/user-attachments/assets/2c994edf-c6d8-4e38-9d2c-879051498af5)

#### **Observation:**
- The box plots for all four price variables show a consistent range of prices with a slight increase in median prices over time.
- The median **Open** and **Close** prices hover around **4500**.
- Outliers exist on the lower end of the price spectrum, with points close to **2500** (which might correspond to periods of market downturns).

#### **Interpretation:**
- The upward trend of the box plots indicates growth in the S&P 500 index over the analyzed period.
- Outliers on the lower end could represent significant market shocks or corrections, possibly due to macroeconomic factors like **recessions** or **geopolitical events**.

### **3. Time Series Line Plot (Close Price)**
We generated a time series plot to track the S&P 500's closing prices over the entire dataset period. The line chart shows the general trajectory of the index, capturing its fluctuations.
![image](https://github.com/user-attachments/assets/638483a4-8cbd-4adb-8bc9-73aa02dee23d)

#### **Observation:**
- From **2019** to early **2020**, the market appears relatively stable, but there is a sharp drop in early **2020**, likely due to the **COVID-19 pandemic**.
- Following this dip, the market begins to recover rapidly, reaching new highs in **2021-2024**, with some occasional dips along the way.
- The closing price of **5381.00** was recorded on **August 9, 2024**.

#### **Interpretation:**
- The significant drop in **2020**, followed by a rapid recovery, mirrors real-world events where global markets, including the S&P 500, were deeply affected by the COVID-19 pandemic.
- The trend upward after **2021** could be attributed to a post-recovery period where the economy begins to stabilize, and market valuations increase.

The **Exploratory Data Analysis (EDA)** provides a strong foundation for understanding the S&P 500's behavior over the five-year period. The descriptive statistics highlight both the stability and volatility in the market, while the visualizations reveal growth trends and market corrections.

---

## **Model Development (LR)**
A **Linear Regression Model** was developed using the `sklearn` library. The following steps were followed to prepare and train the model:

1. **Data Preparation**:
   - The features **Open**, **High**, **Low**, and **Volume** were selected as input (**X**).
   - The **Close** price was set as the target variable (**Y**).

2. **Train-Test Split**:
   - The dataset was split into a training set (**70%**) and a test set (**30%**).
   ![image](https://github.com/user-attachments/assets/6034ce93-336e-46cd-a157-aab6721abe7f)

3. **Model Training**:
   - A **Linear Regression model** was created and trained on the training dataset.
   ![image](https://github.com/user-attachments/assets/08e35557-22b1-4f3a-b88f-2ebc441c5fa2)

4. **Predictions**:
   - The model made predictions on both the training and testing datasets.
   ![image](https://github.com/user-attachments/assets/a4266a1d-9ee9-4c8b-9a43-e1a4a3eafa8c)

## **Results (LR)**
The model’s performance was evaluated using various metrics, which give insight into how well the model performs on the test data:

- **Mean Squared Error (MSE)**: **1.3967** – Measures the average of the squares of the errors. Lower values indicate a better fit.
- **Root Mean Squared Error (RMSE)**: **1.1818** – The square root of MSE. It provides an error metric in the same units as the data.
- **Mean Absolute Error (MAE)**: **0.7173** – Measures the average of the absolute errors between predicted and actual values. Lower values mean better accuracy.
- **R² Score**: **0.9999** – Represents how well the model explains the variability of the target. A score close to 1 indicates an excellent fit.

These metrics demonstrate that the model performs exceptionally well, with an **R² score** of almost **1**, indicating that the model explains nearly all of the variance in the S&P 500 closing prices. However, it is crucial to note that stock prices are volatile and influenced by many external factors, and further validation with real-world data would be necessary.

---

## **Financial Analysis (LR)**
The extremely high **R² score (0.9999)** indicates that the linear regression model is very accurate in explaining the relationships between the input features (**Open**, **High**, **Low**, **Volume**) and the target (**Close**) prices. However, such accuracy should be interpreted carefully:

### **1. Financial Implication:**
While the model demonstrates an excellent fit on the historical data, it is important to understand that stock markets are driven by both quantitative and qualitative factors. News events, global economic trends, and investor sentiment can heavily impact market prices, none of which are included in this model.

### **2. Risk and Predictive Power:**
The **RMSE** and **MAE** provide an idea of the deviation of the predictions from actual prices. In financial terms, these deviations are minimal (around 1 point). This level of error would be acceptable in many trading scenarios, given the typical volatility of the S&P 500 index.

### **3. Overfitting Consideration:**
The near-perfect **R² score** raises a red flag for overfitting. Although the model performs well on the test set, it may not generalize as well to unseen data. In finance, markets are influenced by many unpredictable factors, so over-reliance on historical patterns could lead to misleading predictions in the future.

---

### **Overall Analysis:**
This model could be useful in understanding trends and making predictions based on historical data but should be combined with other financial analysis methods for real-world decision-making.

## **Conclusion (LR)**
The **Linear Regression model** provides a basic approach to predict stock prices using historical S&P 500 data. While the model demonstrates high accuracy on the dataset, stock market prices are influenced by a vast range of factors. Thus, the model should not be used as a sole predictor for trading or investment decisions.

### **Future Improvements**:
- Incorporating external variables such as **market indicators**, **financial news sentiment**, or more advanced models like **time series analysis** or **neural networks** for better predictions.

---

# **S&P 500 Stock Market Prediction - Using Long Short-Term Memory (LSTM)**

## **Project Overview (LSTM)**
This project aims to predict the **S&P 500 index's closing stock prices** using a **Long Short-Term Memory (LSTM)** model, a specialized type of recurrent neural network (RNN) that excels at learning from sequential data. The project utilizes **five years' worth of S&P 500 data**, split into training and testing sets. The goal is to evaluate how well the LSTM model can forecast **future stock prices**, specifically predicting the next **10 days** based on past performance.

---

## **Dataset Information (LSTM)**
The dataset comprises **five years of stock prices**, including the following key columns:
- **Date**: The specific timestamp of the stock price.
- **Open**: Opening price of the S&P 500 index.
- **High**: The highest price recorded during the period.
- **Low**: The lowest price recorded.
- **Close**: The closing price of the index (target for prediction).
- **Volume**: Total volume of stocks traded during the time frame.

The data is preprocessed by extracting the 'Close' prices for modeling purposes.
![image](https://github.com/user-attachments/assets/72954448-8d02-484e-9b72-b63b45342785)

---

## **Model Architecture Long Short-Term Memory (LSTM)**

The model is designed with **two layers of LSTM units**, each containing **50 units**. Additionally, **Dropout layers** are included to prevent overfitting, dropping 20-30% of units at random.

### **Model Structure (LSTM)**
- **First LSTM layer**: 50 units, returns sequences for the next layer.
- **Second LSTM layer**: 50 units, no return sequences.
- **Dropout layers**: Applied after each LSTM layer to improve generalization.
- **Dense layer**: Final output layer for predicting the stock price.
![image](https://github.com/user-attachments/assets/547dbe46-7774-4c1b-b7ca-e862c3a020af)

The model is compiled using the **Adam optimizer** and the **mean squared error (MSE)** loss function, which is suitable for regression problems like stock price prediction.

---

## **Training Process (LSTM)**
The model is trained over **10 epochs** with a batch size of **64**, using the **MinMaxScaler** to scale data between 0 and 1 for efficient learning. Training loss was minimized over time, indicating the model's ability to learn and adjust predictions.
![image](https://github.com/user-attachments/assets/09c5e120-e782-4cf6-ad60-654a48945997)

---

## **Model Evaluation Metrics (LSTM)**
The model's performance was evaluated using four key metrics:
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **R-squared (R²)**

These metrics are essential in understanding the model's accuracy, precision, and ability to predict future price movements in the S&P 500.

---

### **Training Metrics:**
- **Training RMSE**: 31.00
- **Training R²**: 0.997

### **Test Metrics:**
- **Test RMSE**: 68.88
- **Test MAE**: 52.81
- **Test R²**: 0.970

---

## **Detailed Analysis of the Metrics**

### **1. Mean Squared Error (MSE):**
MSE measures the average of the squared differences between the predicted and actual values. 
- **Training MSE**: 31.00
- **Test MSE**: 68.88

#### **Financial Implications:**
- The low MSE on the training data shows that the model performed very well on the data it learned from. However, the higher MSE on the test data suggests some difficulty generalizing to unseen data.
- **In finance**: The small error on the training set indicates that the model is capturing past trends effectively. However, the higher error on the test set may reflect real-world complexities like **market volatility** or external shocks (e.g., economic news or geopolitical events).

---

### **2. Root Mean Squared Error (RMSE):**
RMSE is the square root of MSE and represents the error in the same unit as the predicted variable (stock prices).
- **Training RMSE**: 31.00
- **Test RMSE**: 68.88

#### **Financial Implications:**
- An **RMSE of 68.88** means that on average, the predicted stock prices deviate from the actual stock prices by about 68.88 points. 
- **In finance**: For an investor or trader, an RMSE of 68.88 points suggests that while the model can generally follow the trend, it may miss some **short-term price movements** or fail to capture sudden spikes or dips.

---

### **3. Mean Absolute Error (MAE):**
MAE measures the average magnitude of the errors in predictions without considering their direction.
- **Test MAE**: 52.81

#### **Financial Implications:**
- An MAE of 52.81 means that on average, the model's predictions are off by around **52.81 points** from the actual stock prices.
- **In finance**: For short-term traders, this error could mean missing price action by a substantial margin. However, for longer-term market predictions, this error margin is relatively low compared to the general **volatility of the S&P 500**.

---

### **4. R-Squared (R²):**
R² measures how well the model explains the variance in the actual data.
- **Training R²**: 0.997
- **Test R²**: 0.970

#### **Financial Implications:**
- **Test R² of 0.970** shows that the model explains 97% of the variance in the test data.
- **In finance**: This is an excellent score for financial forecasting. It provides confidence that the model captures **market dynamics** well and can be used as a **decision-support tool** for investment strategies.

---

## **Predicted vs Actual Stock Prices Financial Implications**
The model's ability to predict future stock prices was tested by forecasting the next **10 days** based on past price trends. The small green line in the graph indicates these future predictions.
![image](https://github.com/user-attachments/assets/73ad3a8a-9eba-4df8-8a2d-51fb2133f58b)


### **Future Prediction Analysis:**
1. **Trend Continuation**: The model forecasts a continuation of the upward trend, mirroring the historical price movements of the S&P 500.
2. **Market Correction Possibility**: The small dip observed in the predicted future prices indicates the model anticipates a potential **market correction**.
3. **Volatility Consideration**: The model does not appear to capture sudden, sharp movements due to the smoothing effect of LSTM, making it better suited for **long-term investors** rather than short-term traders.

---

## **Financial Insights Based on the Model Evaluation**

### **For Investors:**
- **R² of 0.970** provides strong confidence that the model can predict general market trends with a high degree of accuracy.
- **RMSE and MAE** values suggest caution during times of **high volatility**.

### **For Traders:**
- The **Test RMSE of 68.88** may be concerning for short-term traders who need higher precision in predictions.
- The model’s smooth price prediction curve could miss capturing **volatile spikes** or **sudden drops**, making it less ideal for **high-frequency trading**.

### **Investment Strategy Consideration:**
- **Long-term strategy**: The model is suitable for **buy-and-hold strategies**, as it reliably predicts trends over longer periods.
- **Risk Management**: Investors should use additional tools, like **stop-loss orders** or **hedging strategies**, to protect against unexpected market events that the model cannot predict.

---

## **Conclusion (LSTM)**
In conclusion, the **LSTM model** provides a powerful tool for predicting **long-term trends** in the S&P 500. It exhibits strong predictive power with an **R² of 0.970** and relatively low error margins. However, it faces challenges in fully anticipating market volatility and **sudden price changes**.

For **long-term investors**, this model offers valuable insights into where the market may be headed. Short-term traders should approach these predictions with caution and combine them with more **granular analysis tools** for volatile conditions.
