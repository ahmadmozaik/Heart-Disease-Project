# ❤️ Heart Disease Prediction Project

This is an end-to-end machine learning project to predict the likelihood of heart disease based on patient data. The project involves data cleaning, exploratory data analysis, feature selection, model training, hyperparameter tuning, and deployment as a web application using Streamlit.

## 🚀 Features
- **Data Preprocessing**: Handles missing values, encodes categorical features, and scales numerical data.
- **Feature Selection**: Uses Random Forest to identify the most predictive features.
- **Model Training**: Compares multiple classification models (Logistic Regression, SVM, Random Forest, Decision Tree).
- **Hyperparameter Tuning**: Optimizes the best model using `GridSearchCV`.
- **Web Application**: An interactive UI built with Streamlit for real-time predictions.

## 📂 File Structure
Heart_Disease_Project/
├── data/
│   ├── heart_disease.csv
│   └── cleaned_heart_disease.csv
├── models/
│   ├── final_model.pkl
│   └── scaler.pkl
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   └── 06_hyperparameter_tuning.ipynb
├── ui/
│   └── app.py
├── README.md
└── requirements.txt

## ⚙️ How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd Heart-Disease-Project
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run ui/app.py
    ```