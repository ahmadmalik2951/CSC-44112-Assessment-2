# FinTech Fraud Detection: Imbalanced Data Classification using Ensemble Models

## 📌 Project Overview
This repository contains a complete end-to-end Machine Learning pipeline designed to detect malicious financial activity within a heavily imbalanced dataset. The project evaluates and compares the performance of baseline bagging techniques (Random Forest) against advanced gradient boosting architectures (XGBoost) to intercept fraudulent mobile money transactions.

The core commercial challenge addressed in this project is the **Precision-Recall Trade-off** and the **Accuracy Paradox** inherent in anomaly detection (where fraud constitutes only 0.13% of total transactions).

## 📊 Dataset: PaySim
The project utilizes the **PaySim** dataset, a synthetically generated but highly realistic log of mobile money transactions. 

⚠️ **Important Note on Data Access:** Due to GitHub's strict 100 MB file size limit, the raw dataset (`PS_20174392719_1491204439457_log.csv`) could not be uploaded to this repository. 
* To run this code locally, please download the dataset directly from Kaggle: [PaySim Synthetic Financial Datasets for Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1)
* Once downloaded, place the `.csv` file in the root directory of this project before executing the notebook.

## 🛠️ Methodology & Architecture
1. **Exploratory Data Analysis (EDA):** * Proved severe class imbalance (0.13% minority class).
   * Discovered behavioral bottlenecks (fraud strictly confined to `TRANSFER` and `CASH_OUT` types).
   * Engineered chronobiological features (`hour_of_day`) to capture nocturnal malicious cycles.
2. **Data Preprocessing:** Handled categorical variables via One-Hot Encoding and implemented stratified Train/Test splitting to preserve the minority class distribution.
3. **Model Development:**
   * **Baseline:** Unweighted Random Forest Classifier.
   * **Advanced Engine:** eXtreme Gradient Boosting (XGBoost) utilizing the `scale_pos_weight` hyperparameter for algorithmic class weighting.
4. **Hyperparameter Tuning:** Executed `GridSearchCV` on a 10% stratified sample to optimize tree complexity (`max_depth`) and learning step size (`learning_rate`) while prioritizing the F1-Score over standard accuracy.

## 🚀 Key Findings & Results

* **The "Account Emptying" Typology:** Explainable AI (XAI) feature importance extraction confirmed that the origin account's initial balance (`oldbalanceOrg`) and resulting balance (`newbalanceOrig`) were the strongest predictive determinants, proving that malicious actors systematically drain compromised accounts to exactly zero.
* **The Precision-Recall Dilemma:** * The **Random Forest** baseline achieved high precision (0.97) but missed 20% of actual fraud (Recall: 0.80).
  * The **Optimized XGBoost** architecture achieved near-perfect fraud interception (Recall: 0.99) but suffered a severe precision collapse (0.33) due to the aggressive algorithmic weighting, resulting in a 67% false-positive rate.
* **Computational Efficiency (ILO2):** XGBoost demonstrated vastly superior computational suitability for high-volume banking. While a single unconstrained Random Forest took **9 minutes** to train, the XGBoost architecture (aided by shallow tree limits and hardware optimization) executed a rigorous cross-validated Grid Search in merely **22 seconds**.

## 💻 Tech Stack
* **Language:** Python 3.x
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost
* **Visual Analytics:** Matplotlib, Seaborn

## ⚙️ How to Run the Project
1. Clone this repository to your local machine.
2. Download the dataset from Kaggle (link above) and place it in the same directory as the Jupyter Notebook/Python script.
3. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn
4. Run all cells in the provided Jupyter Notebook (.ipynb file) or execute the Python script.
