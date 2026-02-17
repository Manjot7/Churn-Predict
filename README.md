# Bank Customer Churn Prediction System

A machine learning application for predicting customer churn using 15 different algorithms and advanced feature engineering.

## Live Demo

[View Live Application](https://manjot7-churn-predict.streamlit.app)

## Features

* 15 machine learning models including XGBoost, LightGBM, CatBoost
* Advanced feature engineering with 15 new features
* Class imbalance handling using SMOTE-Tomek
* Hyperparameter optimization via GridSearchCV
* Interactive prediction interface
* Comprehensive model performance analysis
* Real-time churn predictions

## Technology Stack

* Python 3.9
* Scikit-Learn
* XGBoost, LightGBM, CatBoost
* Streamlit
* Pandas, NumPy
* Imbalanced-Learn

## Dataset

* Source: Kaggle Bank Customer Churn Dataset
* 10,000 customer records
* 26 features (11 original + 15 engineered)
* Binary target variable (Churn / Not Churn)
* Class imbalance ratio of 4:1

## Model Performance

Best performing model (Voting Ensemble) achieves:

| Metric | Value |
|--------|-------|
| Accuracy | 85.93% |
| Precision | 70.77% |
| Recall | 52.70% |
| F1-Score | 0.60 |
| ROC-AUC | 0.87 |
| Matthews Correlation Coefficient | 0.53 |
| Cohen's Kappa | 0.52 |

## Models Evaluated

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gradient Boosting Classifier
5. Support Vector Machine
6. K-Nearest Neighbors
7. Gaussian Naive Bayes
8. AdaBoost Classifier
9. Extra Trees Classifier
10. Linear Discriminant Analysis
11. Quadratic Discriminant Analysis
12. XGBoost
13. LightGBM
14. CatBoost
15. Voting Ensemble

## Project Structure
```
churn-prediction/
├── .streamlit/
│   └── config.toml
├── app.py
├── customer_churn_prediction_enhanced.py
├── churn.csv
├── requirements.txt
├── README.md
├── .gitignore
├── best_model.pkl
├── scaler.pkl
├── label_encoders.pkl
├── model_comparison_results.csv
├── model_results_summary.txt
├── eda_comprehensive_analysis.png
├── model_evaluation_comprehensive.png
└── roc_curves_top_models.png
```

## Usage

### Application Pages

1. **Home** - Project overview and key performance metrics
2. **Model Performance** - Detailed model comparison and evaluation charts
3. **Make Prediction** - Real-time churn prediction for individual customers
4. **Exploratory Analysis** - Interactive dataset exploration with filters
5. **About** - Detailed project and methodology information

### Running Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Contact

* GitHub: [@manjot7](https://github.com/manjot7)
* LinkedIn: [LinkedIn Profile](https://linkedin.com/in/manjot7)

## License

This project is open source and available under the MIT License.
