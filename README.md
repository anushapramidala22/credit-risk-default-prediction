# Credit Risk Default Prediction

This repository contains the complete analysis and key deliverables for a credit risk modeling project. The primary objective was to analyze historical loan application data to build a predictive model that assesses the probability of a customer defaulting on a loan.

The project goes beyond simple prediction, incorporating advanced data science techniques and culminating in a user-friendly decision-support tool designed for loan officers.

---

### Key Features & Methodology

This project demonstrates a comprehensive, end-to-end data science workflow:

* **Robust Data Preprocessing:** Handled missing data (including empty strings and `NA` values) and anomalous data (e.g., illogical dates) using advanced imputation (`mice` package in R).
* **In-depth Exploratory Data Analysis (EDA):** Generated key visualizations to understand customer demographics, risk patterns, and the concentration of financial risk within different segments.
* **Advanced Model Validation:** Employed **10-fold Cross-Validation** to ensure the model's performance is stable and reliable, moving beyond a simple train/test split.
* **Imbalanced Data Handling:** Implemented the **SMOTE** (Synthetic Minority Over-sampling Technique) to address the class imbalance inherent in the dataset. This resulted in a model that is significantly better at identifying the rare but critical default events, prioritizing business value over simple accuracy.
* **Strategic Segment Analysis:** Performed deep-dive analysis on model performance for "grey area" and "high-value" customer segments to identify specific business risks and formulate targeted policies.
* **Interactive Decision-Support Tool:** Developed a user-friendly GUI with Streamlit that includes:
    * Real-time risk scoring.
    * Built-in data validation to prevent future data quality issues.
    * Nuanced, context-aware recommendations for loan officers.
    * Customer profiling to provide explainability and build user trust.

---

### Repository Contents

This repository includes the following key files:

* `PD_Model.R`: The complete R script containing the entire data analysis pipeline, from data cleaning and EDA to model building (with SMOTE) and strategic analysis.
* `app.py`: The Python script for the interactive Streamlit GUI.
* `PD_Calculator.xlsx`: A user-friendly Excel-based tool that implements the final model's formula.

**Note:** Due to confidentiality, the raw `logreg_data.csv` file is not included in this public repository.

---

### How to Run the Interactive GUI

To run the Streamlit decision-support tool on your local machine, please follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/credit-risk-default-prediction.git](https://github.com/your-username/credit-risk-default-prediction.git)
    cd credit-risk-default-prediction
    ```

2.  **Install Dependencies:**
    Make sure you have Python 3 installed. Then, run the following command in your terminal to install the necessary libraries:
    ```bash
    pip3 install streamlit pandas
    ```

3.  **Run the App:**
    Once the dependencies are installed, run the following command:
    ```bash
    streamlit run app.py
    ```

Your web browser should automatically open with the application running.

