import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    h1 {
        color: #1f2937;
        font-weight: 600;
    }
    h2 {
        color: #374151;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

# Application title
st.title("Bank Customer Churn Prediction System")
st.markdown("""
This application provides machine learning based predictions for customer churn analysis.
The system uses advanced algorithms and feature engineering to deliver accurate predictions.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Home", "Model Performance", "Make Prediction", "Exploratory Analysis", "About"]
)

# HOME PAGE
if page == "Home":
    st.header("Project Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Models",
            value="15",
            help="Number of machine learning models evaluated"
        )
    
    with col2:
        st.metric(
            label="Best Accuracy",
            value="85.9%",
            help="Highest accuracy achieved across all models"
        )
    
    with col3:
        st.metric(
            label="Best ROC AUC",
            value="0.87",
            help="Area under the ROC curve for the best performing model"
        )
    
    with col4:
        st.metric(
            label="Features",
            value="26",
            help="Total number of features including engineered ones"
        )
    
    st.markdown("---")
    
    # Project highlights
    st.subheader("Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Machine Learning Models**
        * Logistic Regression
        * Decision Tree Classifier
        * Random Forest Classifier
        * Gradient Boosting Classifier
        * Support Vector Machine
        * K Nearest Neighbors
        * Gaussian Naive Bayes
        * AdaBoost Classifier
        * Extra Trees Classifier
        * Linear Discriminant Analysis
        * Quadratic Discriminant Analysis
        * XGBoost (optional)
        * LightGBM (optional)
        * CatBoost (optional)
        * Voting Ensemble
        """)
    
    with col2:
        st.markdown("""
        **Technical Capabilities**
        * Advanced feature engineering with 15 new features
        * Class imbalance handling using SMOTE Tomek
        * Hyperparameter optimization via GridSearchCV
        * Cross validation for robust evaluation
        * Ensemble methods for improved predictions
        * Comprehensive model comparison
        * Real time prediction interface
        * Interactive data exploration
        """)
    
    st.markdown("---")
    
    # Technology stack
    st.subheader("Technology Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Core Libraries**
        * Python 3.8+
        * NumPy
        * Pandas
        * Scikit Learn
        """)
    
    with col2:
        st.markdown("""
        **ML Frameworks**
        * XGBoost
        * LightGBM
        * CatBoost
        * Imbalanced Learn
        """)
    
    with col3:
        st.markdown("""
        **Visualization**
        * Matplotlib
        * Seaborn
        * Streamlit
        """)
    
    st.markdown("---")
    
    # Dataset information
    st.subheader("Dataset Information")
    
    dataset_info = pd.DataFrame({
        'Attribute': ['Source', 'Total Samples', 'Original Features', 'Engineered Features', 
                     'Target Variable', 'Class Distribution', 'Data Quality'],
        'Description': ['Kaggle Bank Customer Churn Dataset', '10,000', '11', '15', 
                       'Binary (Churn / Not Churn)', 'Imbalanced (4:1 ratio)', 'No missing values']
    })
    
    st.table(dataset_info)

# MODEL PERFORMANCE PAGE
elif page == "Model Performance":
    st.header("Model Performance Analysis")
    
    try:
        # Load model results
        results_df = pd.read_csv('model_comparison_results.csv', index_col=0)
        
        # Display performance table
        st.subheader("Comprehensive Model Comparison")
        
        # Format the dataframe
        formatted_df = results_df.copy()
        for col in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(
            formatted_df.style.set_properties(**{'text-align': 'center'}),
            use_container_width=True
        )
        
        # Identify best model
        best_model_name = results_df['roc_auc'].idxmax()
        best_roc_auc = results_df.loc[best_model_name, 'roc_auc']
        
        st.success(f"Best Performing Model: {best_model_name} (ROC AUC: {best_roc_auc:.4f})")
        
        st.markdown("---")
        
        # Visualization section
        st.subheader("Performance Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**ROC AUC Score Comparison**")
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            
            sorted_roc = results_df['roc_auc'].sort_values()
            colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_roc)))
            
            ax1.barh(range(len(sorted_roc)), sorted_roc.values, color=colors)
            ax1.set_yticks(range(len(sorted_roc)))
            ax1.set_yticklabels(sorted_roc.index)
            ax1.set_xlabel('ROC AUC Score')
            ax1.set_title('Model Performance by ROC AUC')
            ax1.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig1)
        
        with col2:
            st.markdown("**Accuracy vs F1 Score Analysis**")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            
            scatter = ax2.scatter(
                results_df['accuracy'], 
                results_df['f1_score'], 
                s=150, 
                alpha=0.6,
                c=range(len(results_df)),
                cmap='viridis'
            )
            
            for idx, model in enumerate(results_df.index):
                ax2.annotate(
                    model, 
                    (results_df['accuracy'].iloc[idx], results_df['f1_score'].iloc[idx]),
                    fontsize=8,
                    alpha=0.7
                )
            
            ax2.set_xlabel('Accuracy')
            ax2.set_ylabel('F1 Score')
            ax2.set_title('Accuracy vs F1 Score Trade Off')
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig2)
        
        st.markdown("---")
        
        # Detailed visualizations
        st.subheader("Detailed Analysis")
        
        tab1, tab2, tab3 = st.tabs([
            "Exploratory Data Analysis", 
            "Model Evaluation", 
            "ROC Curves"
        ])
        
        with tab1:
            try:
                st.image(
                    'eda_comprehensive_analysis.png', 
                    caption='Comprehensive Exploratory Data Analysis',
                    use_column_width=True
                )
            except FileNotFoundError:
                st.warning("EDA visualization not found. Please run the training script.")
        
        with tab2:
            try:
                st.image(
                    'model_evaluation_comprehensive.png', 
                    caption='Comprehensive Model Evaluation Metrics',
                    use_column_width=True
                )
            except FileNotFoundError:
                st.warning("Model evaluation visualization not found.")
        
        with tab3:
            try:
                st.image(
                    'roc_curves_top_models.png', 
                    caption='ROC Curves for Top Performing Models',
                    use_column_width=True
                )
            except FileNotFoundError:
                st.warning("ROC curves visualization not found.")
    
    except FileNotFoundError:
        st.error("Model results file not found. Please run the training script first to generate model_comparison_results.csv")
    except Exception as e:
        st.error(f"An error occurred while loading model performance data: {str(e)}")

# PREDICTION PAGE
elif page == "Make Prediction":
    st.header("Customer Churn Prediction")
    
    st.markdown("Enter customer information to predict churn probability and risk level.")
    
    # Input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Customer Profile")
            credit_score = st.number_input(
                "Credit Score",
                min_value=300,
                max_value=850,
                value=650,
                help="Customer credit score (300 to 850)"
            )
            
            geography = st.selectbox(
                "Geography",
                options=["France", "Germany", "Spain"],
                help="Customer location"
            )
            
            gender = st.selectbox(
                "Gender",
                options=["Male", "Female"],
                help="Customer gender"
            )
            
            age = st.slider(
                "Age",
                min_value=18,
                max_value=100,
                value=40,
                help="Customer age in years"
            )
        
        with col2:
            st.subheader("Account Details")
            tenure = st.slider(
                "Tenure (Years)",
                min_value=0,
                max_value=10,
                value=5,
                help="Years with the bank"
            )
            
            balance = st.number_input(
                "Account Balance ($)",
                min_value=0.0,
                max_value=300000.0,
                value=75000.0,
                step=1000.0,
                help="Current account balance"
            )
            
            num_products = st.selectbox(
                "Number of Products",
                options=[1, 2, 3, 4],
                help="Number of bank products"
            )
            
            estimated_salary = st.number_input(
                "Estimated Salary ($)",
                min_value=0.0,
                max_value=200000.0,
                value=100000.0,
                step=1000.0,
                help="Estimated annual salary"
            )
        
        with col3:
            st.subheader("Banking Behavior")
            has_cr_card = st.selectbox(
                "Has Credit Card",
                options=["Yes", "No"],
                help="Whether customer has a credit card"
            )
            
            is_active = st.selectbox(
                "Is Active Member",
                options=["Yes", "No"],
                help="Whether customer is actively using services"
            )
        
        submit_button = st.form_submit_button("Generate Prediction", type="primary")
    
    if submit_button:
        # Create feature dictionary
        input_data = {
            'CreditScore': credit_score,
            'Geography': geography,
            'Gender': gender,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_products,
            'HasCrCard': 1 if has_cr_card == "Yes" else 0,
            'IsActiveMember': 1 if is_active == "Yes" else 0,
            'EstimatedSalary': estimated_salary
        }
        
        # Feature engineering
        input_data['AgeGroup'] = int(pd.cut(
            [age], 
            bins=[0, 30, 40, 50, 60, 100], 
            labels=[1, 2, 3, 4, 5]
        )[0])
        
        input_data['BalanceGroup'] = int(pd.cut(
            [balance], 
            bins=[-0.01, 0.01, 50000, 100000, 150000, 300000], 
            labels=[0, 1, 2, 3, 4]
        )[0])
        
        input_data['CreditScoreCategory'] = int(pd.cut(
            [credit_score], 
            bins=[0, 580, 670, 740, 800, 900], 
            labels=[1, 2, 3, 4, 5]
        )[0])
        
        input_data['TenureGroup'] = int(pd.cut(
            [tenure], 
            bins=[-0.01, 2, 5, 10, 15], 
            labels=[1, 2, 3, 4]
        )[0])
        
        input_data['BalanceToSalaryRatio'] = balance / estimated_salary if estimated_salary > 0 else 0
        input_data['ProductsPerTenure'] = num_products / tenure if tenure > 0 else num_products
        input_data['IsSenior'] = 1 if age > 60 else 0
        input_data['HasZeroBalance'] = 1 if balance == 0 else 0
        input_data['ActiveWithCard'] = input_data['IsActiveMember'] * input_data['HasCrCard']
        input_data['EngagementScore'] = (
            input_data['IsActiveMember'] * 0.4 + 
            input_data['HasCrCard'] * 0.2 + 
            (num_products / 4) * 0.4
        )
        
        # Additional engineered features
        input_data['CustomerValueScore'] = (
            (balance / 300000) * 0.5 +
            (estimated_salary / 200000) * 0.5
        )
        
        input_data['ActivityScore'] = (
            input_data['IsActiveMember'] * 0.5 +
            (tenure / 10) * 0.3 +
            (num_products / 4) * 0.2
        )
        
        input_data['CreditToAgeRatio'] = credit_score / age
        
        input_data['IsHighValueCustomer'] = 1 if (balance > 127644 and estimated_salary > 149388) else 0
        
        input_data['HasMultipleProducts'] = 1 if num_products > 1 else 0
        
        # Encode categorical variables
        geography_map = {'France': 0, 'Germany': 1, 'Spain': 2}
        gender_map = {'Female': 0, 'Male': 1}
        input_data['Geography'] = geography_map[geography]
        input_data['Gender'] = gender_map[gender]
        
        # Create DataFrame with proper column order
        expected_columns = [
            'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
            'AgeGroup', 'BalanceGroup', 'CreditScoreCategory', 'TenureGroup',
            'BalanceToSalaryRatio', 'ProductsPerTenure', 'IsSenior',
            'HasZeroBalance', 'ActiveWithCard', 'EngagementScore',
            'CustomerValueScore', 'ActivityScore', 'CreditToAgeRatio',
            'IsHighValueCustomer', 'HasMultipleProducts'
        ]
        
        input_df = pd.DataFrame([input_data])
        input_df = input_df[expected_columns]
        
        try:
            # Load model and scaler
            model = pickle.load(open('best_model.pkl', 'rb'))
            scaler = pickle.load(open('scaler.pkl', 'rb'))
            
            # Scale features
            input_scaled = scaler.transform(input_df)
            
            # Generate prediction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.error("**Status:** High Churn Risk")
                else:
                    st.success("**Status:** Low Churn Risk")
            
            with col2:
                st.metric(
                    "Churn Probability",
                    f"{probability[1]*100:.2f}%"
                )
            
            with col3:
                st.metric(
                    "Retention Probability",
                    f"{probability[0]*100:.2f}%"
                )
            
            # Risk visualization
            st.subheader("Risk Assessment Visualization")
            
            fig, ax = plt.subplots(figsize=(12, 2))
            
            ax.barh([0], [probability[1]], color='#ef4444', alpha=0.8, label='Churn Risk')
            ax.barh([0], [probability[0]], left=[probability[1]], color='#10b981', alpha=0.8, label='Retention')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
            ax.set_xlabel('Probability', fontsize=12)
            ax.set_title(f'Churn Risk Analysis: {probability[1]*100:.2f}%', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(axis='x', alpha=0.3)
            
            for i, (prob, label) in enumerate([(probability[1], 'Churn'), (probability[0], 'Retain')]):
                if prob > 0.1:
                    x_pos = probability[1]/2 if i == 0 else probability[1] + probability[0]/2
                    ax.text(x_pos, 0, f'{prob*100:.1f}%', 
                           ha='center', va='center', fontsize=11, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Recommendations
            st.markdown("---")
            st.subheader("Actionable Recommendations")
            
            if prediction == 1:
                st.markdown("""
                **High Risk Customer: Immediate Intervention Required**
                
                * **Priority Action:** Schedule personal consultation with account manager within 48 hours
                * **Retention Strategy:** Offer customized product bundle with preferential rates
                * **Engagement Plan:** Implement loyalty rewards program with quarterly benefits review
                * **Communication:** Send personalized satisfaction survey and follow up on feedback
                * **Product Upgrade:** Present premium banking services with enhanced features
                * **Monitoring:** Add to high priority watch list for monthly relationship reviews
                """)
            else:
                st.markdown("""
                **Low Risk Customer: Maintain and Enhance Relationship**
                
                * **Relationship Management:** Continue current service excellence standards
                * **Cross Selling Opportunity:** Evaluate additional product offerings based on profile
                * **Engagement:** Maintain regular communication through preferred channels
                * **Value Addition:** Provide financial planning resources and market insights
                * **Loyalty Program:** Ensure enrollment in rewards and benefits programs
                * **Feedback Loop:** Annual satisfaction survey to maintain service quality
                """)
            
        except FileNotFoundError:
            st.error("Model files not found. Please ensure best_model.pkl and scaler.pkl are in the application directory.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

# EXPLORATORY ANALYSIS PAGE
elif page == "Exploratory Analysis":
    st.header("Exploratory Data Analysis")
    
    try:
        # Load dataset
        df = pd.read_csv('churn.csv')
        
        # Dataset overview metrics
        st.subheader("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", f"{len(df):,}")
        
        with col2:
            churned = df['Exited'].sum()
            st.metric("Churned Customers", f"{churned:,}")
        
        with col3:
            retained = len(df) - df['Exited'].sum()
            st.metric("Retained Customers", f"{retained:,}")
        
        with col4:
            churn_rate = df['Exited'].mean() * 100
            st.metric("Churn Rate", f"{churn_rate:.2f}%")
        
        st.markdown("---")
        
        # Dataset sample
        st.subheader("Dataset Sample")
        st.dataframe(
            df.head(15),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Statistical summary
        st.subheader("Statistical Summary")
        st.dataframe(
            df.describe(),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Interactive filters
        st.subheader("Interactive Data Exploration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_geography = st.multiselect(
                "Filter by Geography",
                options=df['Geography'].unique().tolist(),
                default=df['Geography'].unique().tolist(),
                help="Select one or more countries"
            )
        
        with col2:
            age_range = st.slider(
                "Age Range",
                min_value=int(df['Age'].min()),
                max_value=int(df['Age'].max()),
                value=(int(df['Age'].min()), int(df['Age'].max())),
                help="Filter customers by age range"
            )
        
        with col3:
            balance_range = st.slider(
                "Balance Range",
                min_value=float(df['Balance'].min()),
                max_value=float(df['Balance'].max()),
                value=(float(df['Balance'].min()), float(df['Balance'].max())),
                step=1000.0,
                help="Filter customers by account balance"
            )
        
        # Apply filters
        filtered_df = df[
            (df['Geography'].isin(selected_geography)) &
            (df['Age'].between(age_range[0], age_range[1])) &
            (df['Balance'].between(balance_range[0], balance_range[1]))
        ]
        
        st.info(f"Filtered Dataset: {len(filtered_df):,} customers ({len(filtered_df)/len(df)*100:.1f}% of total)")
        
        st.markdown("---")
        
        # Visualizations
        st.subheader("Data Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Age Distribution by Churn Status**")
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            
            filtered_df[filtered_df['Exited']==0]['Age'].hist(
                bins=30, alpha=0.6, label='Retained', ax=ax1, color='#10b981'
            )
            filtered_df[filtered_df['Exited']==1]['Age'].hist(
                bins=30, alpha=0.6, label='Churned', ax=ax1, color='#ef4444'
            )
            
            ax1.set_xlabel('Age', fontsize=11)
            ax1.set_ylabel('Frequency', fontsize=11)
            ax1.set_title('Age Distribution', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig1)
        
        with col2:
            st.markdown("**Churn Rate by Geography**")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            
            churn_by_geo = filtered_df.groupby('Geography')['Exited'].mean() * 100
            bars = ax2.bar(churn_by_geo.index, churn_by_geo.values, color='#3b82f6', alpha=0.7)
            
            ax2.set_ylabel('Churn Rate (%)', fontsize=11)
            ax2.set_xlabel('Geography', fontsize=11)
            ax2.set_title('Churn Rate by Geography', fontsize=12, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig2)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**Balance Distribution**")
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            
            filtered_df.boxplot(column='Balance', by='Exited', ax=ax3)
            ax3.set_xlabel('Exited (0=Retained, 1=Churned)', fontsize=11)
            ax3.set_ylabel('Balance ($)', fontsize=11)
            ax3.set_title('Balance Distribution by Churn Status', fontsize=12, fontweight='bold')
            plt.suptitle('')
            
            plt.tight_layout()
            st.pyplot(fig3)
        
        with col4:
            st.markdown("**Churn Rate by Number of Products**")
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            
            churn_by_products = filtered_df.groupby('NumOfProducts')['Exited'].mean() * 100
            bars = ax4.bar(churn_by_products.index, churn_by_products.values, color='#8b5cf6', alpha=0.7)
            
            ax4.set_ylabel('Churn Rate (%)', fontsize=11)
            ax4.set_xlabel('Number of Products', fontsize=11)
            ax4.set_title('Churn Rate by Products', fontsize=12, fontweight='bold')
            ax4.grid(axis='y', alpha=0.3)
            
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig4)
        
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure churn.csv is in the application directory.")
    except Exception as e:
        st.error(f"An error occurred while loading the dataset: {str(e)}")

# ABOUT PAGE
elif page == "About":
    st.header("About This Project")
    
    st.markdown("""
    ### Project Description
    
    This customer churn prediction system is a comprehensive machine learning application designed to 
    identify customers at risk of leaving a banking institution. The system leverages advanced 
    algorithms and feature engineering techniques to provide accurate predictions and actionable insights.
    
    ### Methodology
    
    **Data Processing**
    * Comprehensive exploratory data analysis
    * Feature engineering with 15 additional features
    * Class imbalance handling using SMOTE Tomek technique
    * Feature scaling and normalization
    
    **Model Development**
    * Evaluation of 15 different machine learning algorithms
    * Hyperparameter optimization using GridSearchCV
    * Cross validation for robust performance estimation
    * Ensemble methods for improved accuracy
    
    **Evaluation Metrics**
    * Accuracy and precision scores
    * Recall and F1 score
    * ROC AUC analysis
    * Confusion matrix evaluation
    * Precision recall curves
    
    ### Features
    
    **Original Features**
    * Credit Score
    * Geography
    * Gender
    * Age
    * Tenure
    * Balance
    * Number of Products
    * Has Credit Card
    * Is Active Member
    * Estimated Salary
    
    **Engineered Features**
    * Age Group
    * Balance Group
    * Credit Score Category
    * Tenure Group
    * Balance to Salary Ratio
    * Products per Tenure
    * Is Senior
    * Has Zero Balance
    * Active with Card
    * Engagement Score
    * Customer Value Score
    * Activity Score
    * Credit to Age Ratio
    * Is High Value Customer
    * Has Multiple Products
    
    ### Model Performance
    
    The system evaluates multiple machine learning algorithms and selects the best performing model 
    based on ROC AUC score. All models are trained using consistent preprocessing and evaluation 
    protocols to ensure fair comparison.
    
    ### Technical Implementation
    
    **Backend**
    * Python 3.8+
    * Scikit Learn for traditional ML algorithms
    * XGBoost, LightGBM, CatBoost for gradient boosting
    * Imbalanced Learn for handling class imbalance
    
    **Frontend**
    * Streamlit for web interface
    * Matplotlib and Seaborn for visualizations
    * Interactive components for user input
    
    **Deployment**
    * Containerized application
    * Cloud ready architecture
    * Scalable prediction pipeline
    
    ### Usage Instructions
    
    1. **Home Page**: View project overview and key metrics
    2. **Model Performance**: Analyze model comparison and evaluation results
    3. **Make Prediction**: Input customer data to generate churn predictions
    4. **Exploratory Analysis**: Explore dataset with interactive filters
    
    ### Data Privacy
    
    This application uses synthetic data for demonstration purposes. In production environments, 
    all customer data should be handled according to relevant data protection regulations and 
    organizational privacy policies.
    
    ### Future Enhancements
    
    * Real time model retraining pipeline
    * A/B testing framework for model versions
    * Advanced explainability features (SHAP values, LIME)
    * Integration with CRM systems
    * Automated monitoring and alerting
    * Multi model deployment strategies
    
    ### Technical Support
    
    For technical issues or questions regarding this application, please refer to the project 
    documentation or contact the development team.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem 0; color: #6b7280;'>
    <p style='margin: 0; font-size: 0.9rem;'>
        Customer Churn Prediction System | 
        <a href='https://github.com/Manjot7/Churn-Predict' style='color: #3b82f6; text-decoration: none;'>GitHub Repository</a> | 
        <a href='https://www.linkedin.com/in/manjot-singh-6b5110228/' style='color: #3b82f6; text-decoration: none;'>LinkedIn Profile</a>
    </p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.85rem;'>
        Built with Python, Scikit Learn, and Streamlit
    </p>
</div>
""", unsafe_allow_html=True)