# 🏠 Bangalore House Price Predictor

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange.svg)

An end-to-end machine learning project that predicts real estate prices in Bangalore. It features a robust data cleaning pipeline, multiple regression models, and an interactive web application built with Streamlit.

## 🚀 Live Demo
Check out the live application here: [Bangalore House Price Predictor](LINK_TO_YOUR_STREAMLIT_CLOUD_APP)

## 📊 Key Features
- **Data Cleaning & EDA**: Handles missing values, converts 'total_sqft' ranges, and removes statistical outliers based on location and BHK.
- **Feature Engineering**: Creates a `price_per_sqft` feature and performs one-hot encoding for over 240 Bangalore locations.
- **Modeling**: Compares Linear Regression, Random Forest, and XGBoost. The final Linear Regression model achieves an R² score of **0.686**.
- **Deployment**: A user-friendly web interface built with **Streamlit** and deployed on the cloud.

## 🛠️ Technologies Used
- **Language**: Python 3.9
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost
- **Framework**: Streamlit
- **Model Persistence**: Joblib

## 🗂️ Project Structure
app.py # Streamlit web application
Bangalore_House_Data.csv # Raw dataset
Bangalore_House_Price_Prediction.ipynb # Jupyter notebook with full analysis
linear_regression_model.pkl # Trained model file
requirements.txt # Python dependencies
README.md # Project documentation
app screenshot


## 🧠 Model Development
The model development followed a structured data science pipeline:

1.  **Data Cleaning**: Dropped irrelevant columns (`area_type`, `society`), handled null values, and converted `size` (e.g., "2 BHK") to a numeric `bhk` feature [citation:10].
2.  **Feature Engineering**:
    *   Created `price_per_sqft` to normalize prices.
    *   Applied a custom function to convert range values in `total_sqft` (e.g., "1056 - 1500") to their average.
3.  **Outlier Treatment**:
    *   Removed entries where `total_sqft` per `bhk` was less than 300 sqft.
    *   Removed outliers using a location-wise standard deviation filter on `price_per_sqft` and a BHK-based pricing logic.
4.  **Model Training & Evaluation**:
    *   Encoded the 'location' feature using OneHotEncoder, resulting in over 240 new features.
    *   Trained and evaluated three models, selecting Linear Regression for its balance of performance and simplicity.

## 📈 Results
| Model | R² Score |
| :--- | :--- |
| Linear Regression | 0.686 |
| Random Forest | 0.757 |
| **XGBoost (Best)** | **0.767** |

## 🖥️ Local Setup & Installation
To run this project on your local machine:

1.  **Clone the repository**
    ```bash
    git clone https://github.com/YOUR_USERNAME/Bangalore-House-Price-Prediction.git
    cd Bangalore-House-Price-Prediction
    
2. **Create a virtual environment**
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   
3. **Install dependencies**
   bash
   pip install -r requirements.txt
   
4. **Run the Streamlit app**
   bash
   streamlit run app.py
