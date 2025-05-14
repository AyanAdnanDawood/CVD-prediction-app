import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                           accuracy_score, precision_score,
                           recall_score, f1_score, classification_report,
                           RocCurveDisplay, PrecisionRecallDisplay, roc_curve, auc,
                           precision_recall_curve)
from lime import lime_tabular
from lime.lime_tabular import LimeTabularExplainer
from sklearn.calibration import calibration_curve

# Custom styling
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Updated CSS with optimal background for text visibility
# Custom styling with unique color palette
st.markdown("""
<style>
    /* Main app background - deep teal */
    .main, .stApp {
        background-color: #D3D3D3 !important;
    }
    
    /* Sidebar background - darker teal */
    .css-1d391kg, .sidebar .sidebar-content {
        background-color: #09ab3b !important;
    }
    
    /* Content cards - cream */
    .page-section, 
    .input-section, 
    .metric-card, 
    .prediction-card,
    .feature-importance-plot,
    .model-comparison,
    .lime-plot {
        background-color: #f5f0e8 !important;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #83c5be !important;
    }
    
    /* High risk card - soft coral */
    .high-risk {
        background-color: #ffddd2 !important;
        border-left: 5px solid #e29578 !important;
    }
    
    /* Low risk card - mint green */
    .low-risk {
        background-color: #e2f3cb !important;
        border-left: 5px solid #83c5be !important;
    }
    
    /* Warning card - pale yellow */
    .warning-card {
        background-color: #fff3b0 !important;
        border-left: 5px solid #ffd166 !important;
    }
    
    /* Text elements */
    body, p, h1, h2, h3, h4, h5, h6,
    .st-b7, .st-c0, .st-c1, .st-c2,
    .stMarkdown, .stAlert {
        color: #FF474C !important;
    }
    
    /* Sidebar text - cream */
    .sidebar .sidebar-content * {
        color: #FF474C !important;
    }
    
    /* LIME explanation */
    .lime-plot, 
    .lime-plot .exp-container,
    .lime-plot iframe {
        background-color: #f5f0e8 !important;
        color: #3a3a3a !important;
    }
    
    /* LIME impact colors */
    .lime-plot .exp-feature-impact-positive {
        background-color: #e29578 !important;
    }
    .lime-plot .exp-feature-impact-negative {
        background-color: #83c5be !important;
    }
    
    /* Button styling - coral */
    .stButton>button {
        background-color: #e29578 !important;
        color: #f5f0e8 !important;
        border: none;
    }
    
    /* Progress bar */
    .progress-container {
        background-color: #f5f0e8 !important;
    }
    
    /* Links */
    a {
        color: #006d77 !important;
    }
    
    /* NEW: Table specific styling */
    .stMarkdown table {
        color: #FF0000 !important;
        width: 100%;
    }
    
    .stMarkdown th {
        background-color: #333333 !important;
        color: #FFFFFF !important;
        font-weight: bold !important;
    }
    
    .stMarkdown td {
        background-color: #f5f0e8 !important;
        color: #FF0000 !important;
    }
    
    .highlight-max {
        background-color: #d4f1d4 !important;
        color: #000000 !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

def safe_progress_bar(value):
    """Safely display progress bar with proper value handling"""
    try:
        progress_value = float(value)
        progress_value = max(0.0, min(1.0, progress_value))
        st.progress(progress_value)
    except (ValueError, TypeError) as e:
        st.error(f"Could not display progress bar: {str(e)}")
        st.error(f"Value received: {value} ({type(value)})")

# Load dataset
@st.cache_data
def load_data():
    try:
        # Try multiple possible file paths/names
        try_paths = [
            "CVD_cleaned.csv",
            "heart.csv",
            "heart_disease.csv",
            "data/heart.csv"
        ]

        for path in try_paths:
            try:
                df = pd.read_csv(path)
                st.success(f"Successfully loaded data from {path}")
                return df
            except:
                continue
                
        # If none of the paths worked, try a demo dataset
        from sklearn.datasets import fetch_openml
        heart = fetch_openml(name="heart", version=1, as_frame=True)
        df = heart.frame
        st.warning("Using demo dataset as fallback")
        return df
        
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return pd.DataFrame()

df = load_data()

# Check if data loaded successfully
if df.empty:
    st.error("No data available. Please check your data file.")
    st.stop()

# Preprocessing
try:
    # Try to find the target column (case insensitive)
    target_col = next((col for col in df.columns if 'heart' in col.lower() or 'disease' in col.lower()), None)

    if target_col is None:
        st.error("Could not find target column (Heart Disease) in dataset")
        st.stop()

    X = df.drop(target_col, axis=1)
    y = df[target_col].apply(lambda x: 1 if str(x).lower() in ['yes', '1', 'true'] else 0)

    # Identify categorical and numerical columns
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns
    num_cols = X.select_dtypes(include=['float64', 'int64']).columns

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                      test_size=0.2, 
                                                      random_state=42, 
                                                      stratify=y)

    # Models
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
    }

    pipelines = {}
    for name, model in models.items():
        pipelines[name] = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        pipelines[name].fit(X_train, y_train)

except Exception as e:
    st.error(f"Error during model setup: {str(e)}")
    st.stop()

# Sidebar for navigation
st.sidebar.title("❤ Heart Disease Predictor")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=100)
app_mode = st.sidebar.radio("Navigation", ["Dashboard", "Risk Predictor"])

def find_similar_column(df, possible_names):
    """Helper function to find columns with similar names"""
    for name in possible_names:
        for col in df.columns:
            if name.lower() in col.lower():
                return col
    return None

def show_data_exploration():
    st.title("Data Exploration & Preprocessing")
    st.markdown('<div class="dashboard-title"></div>', unsafe_allow_html=True)
    
    # Overview metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Samples", len(df))
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Heart Disease Cases", f"{y.sum()} ({y.mean():.1%})")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Features Available", len(X.columns))
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    st.subheader("Dataset Overview")
    st.write("First few rows of the dataset:")
    st.dataframe(df.head())
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    st.subheader("Data Distribution")
    
    # Age distribution plot (if age column exists)
    age_col = find_similar_column(df, ['age', 'age_category'])
    if age_col:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, x=age_col, hue=target_col, ax=ax, 
                     order=sorted(df[age_col].unique()))
        ax.set_title('Age Distribution by Heart Disease Status')
        ax.set_xlabel('Age Category')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("Age column not found in dataset")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    st.subheader("Feature Distributions")
    
    # Show distribution of numerical features
    if len(num_cols) > 0:
        num_col = st.selectbox("Select numerical feature to visualize", num_cols)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x=target_col, y=num_col, ax=ax)
        ax.set_title(f'Distribution of {num_col} by Heart Disease Status')
        st.pyplot(fig)
    else:
        st.warning("No numerical features found in dataset")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    st.subheader("Preprocessing Steps")
    st.write("""
    The following preprocessing steps were applied to the data:
    1. Train-Test Split: 80% training, 20% testing (stratified by target variable)
    2. Feature Scaling: Numerical features were standardized (mean=0, std=1)
    3. Categorical Encoding: Categorical features were one-hot encoded
    4. Missing Values: Handled by the respective models during training
    """)
    
    st.write("Preprocessing Pipeline:")
    st.code("""
    ColumnTransformer([
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ])
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def show_model_comparison():
    st.title("Model Comparison: Random Forest vs XGBoost")
    st.markdown('<div class="dashboard-title"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    st.subheader("Model Overview")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        Random Forest
        - Ensemble of decision trees
        - Uses bagging (bootstrap aggregating)
        - Handles non-linear relationships well
        - Less prone to overfitting than single trees
        - Good for medium-sized datasets
        """)
    
    with col2:
        st.markdown("""
        XGBoost
        - Gradient boosted trees
        - Uses boosting (sequential correction of errors)
        - Handles imbalanced data well
        - Includes regularization to prevent overfitting
        - Often achieves higher accuracy
        - Better for large datasets
        """)
    
    st.markdown("""
    Why we use XGBoost:
    - Typically provides better performance with proper tuning
    - Handles missing values internally
    - Offers feature importance metrics
    - More efficient with computational resources
    - Often performs better on structured/tabular data
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Calculate metrics for both models
    model_metrics = []
    for name, pipeline in pipelines.items():
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        model_metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "ROC AUC": roc_auc_score(y_test, y_proba)
        })
    
    metrics_df = pd.DataFrame(model_metrics).set_index("Model")

    # Custom styling function with improved visibility
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: #d4f1d4; color: #000000; font-weight: bold' 
                if v else '' for v in is_max]

    # Display the table with styling
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    st.subheader("Performance Metrics Comparison")
    styled_df = metrics_df.style.format("{:.3f}").apply(highlight_max, axis=0)
    st.dataframe(styled_df)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Plot ROC curves for both models
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    st.subheader("ROC Curve Comparison")
    fig, ax = plt.subplots(figsize=(8, 6))  # Initialize figure and axis here
    
    for name, pipeline in pipelines.items():
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    st.subheader("Precision-Recall Curve Comparison")
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, pipeline in pipelines.items():
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ax.plot(recall, precision, label=f'{name}')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="upper right")
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    st.subheader("Feature Importance Comparison")
    
    # Get feature names after one-hot encoding
    feature_names = (list(num_cols) + 
                    list(pipelines["Random Forest"].named_steps['preprocessor']
                        .named_transformers_['cat']
                        .get_feature_names_out(cat_cols)))
    
    # Plot feature importance for both models
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Random Forest feature importance
    rf_importances = pipelines["Random Forest"].named_steps['classifier'].feature_importances_
    rf_importance = pd.DataFrame({'Feature': feature_names, 'Importance': rf_importances})
    rf_importance = rf_importance.sort_values('Importance', ascending=False).head(10)
    sns.barplot(data=rf_importance, x='Importance', y='Feature', ax=ax1)
    ax1.set_title('Random Forest (Top 10 Feature Importances)')
    
    # XGBoost feature importance
    xgb_importances = pipelines["XGBoost"].named_steps['classifier'].feature_importances_
    xgb_importance = pd.DataFrame({'Feature': feature_names, 'Importance': xgb_importances})
    xgb_importance = xgb_importance.sort_values('Importance', ascending=False).head(10)
    sns.barplot(data=xgb_importance, x='Importance', y='Feature', ax=ax2)
    ax2.set_title('XGBoost (Top 10 Feature Importances)')
    
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

def show_lime_analysis():
    st.title("Model Interpretability with LIME")
    st.markdown('<div class="dashboard-title"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    st.subheader("LIME Explanation")
    st.write("""
    LIME (Local Interpretable Model-agnostic Explanations) is a technique that explains the predictions of any classifier
    by approximating it locally with an interpretable model. It helps understand individual predictions by showing
    how each feature contributed to the prediction for a specific instance.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    model_for_lime = st.selectbox("Select model for LIME explanation", list(pipelines.keys()))
    
    # Get the preprocessor and model from the pipeline
    preprocessor = pipelines[model_for_lime].named_steps['preprocessor']
    model = pipelines[model_for_lime].named_steps['classifier']
    
    # Transform the training data for LIME
    X_train_transformed = preprocessor.transform(X_train)
    
    # Get feature names after preprocessing
    numeric_features = list(num_cols)
    categorical_features = list(cat_cols)
    ohe = preprocessor.named_transformers_['cat']
    ohe_feature_names = ohe.get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(ohe_feature_names)
    
    # Create LIME explainer
    explainer = LimeTabularExplainer(
        training_data=X_train_transformed,
        feature_names=feature_names,
        class_names=['No Heart Disease', 'Heart Disease'],
        mode='classification',
        discretize_continuous=True
    )
    
    # Select a random instance to explain
    sample_idx = np.random.choice(X_test.shape[0], 1)[0]
    X_test_sample = X_test.iloc[[sample_idx]]
    X_test_transformed = preprocessor.transform(X_test_sample)
    
    # Generate LIME explanation
    with st.spinner('Generating LIME explanation...'):
        try:
            exp = explainer.explain_instance(
                data_row=X_test_transformed[0],
                predict_fn=model.predict_proba,
                num_features=10,
                top_labels=1
            )
            
            st.markdown('<div class="lime-plot">', unsafe_allow_html=True)
            st.subheader("LIME Explanation for Selected Case")
            
            # Show the actual prediction
            pred_proba = model.predict_proba(X_test_transformed)[0]
            st.write(f"Model Prediction: {pred_proba[1]:.2%} probability of heart disease")
            
            # Show the actual feature values for this case
            st.write("Feature values for selected case:")
            st.dataframe(X_test_sample)
            
            # Display LIME explanation
            st.write("How each feature contributed to this prediction:")
            
            # Get the explanation as HTML and display it
            html = exp.as_html()
            st.components.v1.html(html, height=800, scrolling=True)
            
            st.markdown("""
            How to read this explanation:
            - The left side shows the features that most influenced the prediction
            - Orange/red bars indicate features that increased the risk prediction
            - Blue bars indicate features that decreased the risk prediction
            - The values show the actual feature values for this case
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error generating LIME explanation: {str(e)}")

if app_mode == "Dashboard":
    dashboard_page = st.sidebar.selectbox(
        "Select Dashboard Page",
        ["Data Exploration", "Model Comparison", "LIME Analysis"]
    )
    
    if dashboard_page == "Data Exploration":
        show_data_exploration()
    elif dashboard_page == "Model Comparison":
        show_model_comparison()
    elif dashboard_page == "LIME Analysis":
        show_lime_analysis()

elif app_mode == "Risk Predictor":
    st.title("Heart Disease Risk Assessment")
    st.markdown("""
    Fill in your health information below to assess your risk of heart disease.
    """)

    col1, col2 = st.columns(2)
    input_data = {}

    with col1:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.subheader("Demographic & Body Measurements")
        
        # Sex/Gender
        sex_col = find_similar_column(df, ['sex', 'gender'])
        if sex_col:
            input_data[sex_col] = st.selectbox("Sex", df[sex_col].unique())
        
        # Age
        age_col = find_similar_column(df, ['age', 'age_category'])
        if age_col:
            if df[age_col].dtype == 'object':
                age_order = ['18-24', '25-29', '30-34', '35-39', '40-44', 
                           '45-49', '50-54', '55-59', '60-64', '65-69', 
                           '70-74', '75-79', '80+']
                available_ages = [age for age in age_order if age in df[age_col].unique()]
                input_data[age_col] = st.selectbox("Age Category", available_ages)
            else:
                input_data[age_col] = st.slider("Age", 
                                               int(df[age_col].min()), 
                                               int(df[age_col].max()), 
                                               int(df[age_col].median()))
        
        # Height (cm)
        if 'Height_(cm)' in X.columns:
            input_data['Height_(cm)'] = st.slider("Height (cm)", 
                                                float(X['Height_(cm)'].min()),
                                                float(X['Height_(cm)'].max()),
                                                float(X['Height_(cm)'].median()))
        
        # Weight (kg)
        if 'Weight_(kg)' in X.columns:
            input_data['Weight_(kg)'] = st.slider("Weight (kg)", 
                                               float(X['Weight_(kg)'].min()),
                                               float(X['Weight_(kg)'].max()),
                                               float(X['Weight_(kg)'].median()))
        
        # Race/Ethnicity
        race_col = find_similar_column(df, ['race', 'ethnicity'])
        if race_col:
            input_data[race_col] = st.selectbox("Race/Ethnicity", df[race_col].unique())
        
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.subheader("Health Conditions")
        
        # Diabetic status
        diabetic_col = find_similar_column(df, ['diabetic', 'diabetes'])
        if diabetic_col:
            input_data[diabetic_col] = st.selectbox("Diabetic Status", df[diabetic_col].unique())
        
        # Depression
        if 'Depression' in X.columns:
            input_data['Depression'] = st.selectbox("History of Depression", 
                                                 X['Depression'].unique())
        
        # Skin Cancer
        if 'Skin_Cancer' in X.columns:
            input_data['Skin_Cancer'] = st.selectbox("History of Skin Cancer",
                                                  X['Skin_Cancer'].unique())
        
        # Other Cancer
        if 'Other_Cancer' in X.columns:
            input_data['Other_Cancer'] = st.selectbox("History of Other Cancer",
                                                   X['Other_Cancer'].unique())
        
        # Arthritis
        if 'Arthritis' in X.columns:
            input_data['Arthritis'] = st.selectbox("History of Arthritis",
                                                X['Arthritis'].unique())
        
        # General health
        health_col = find_similar_column(df, ['gen_health', 'health', 'general_health'])
        if health_col:
            health_order = ['Poor', 'Fair', 'Good', 'Very good', 'Excellent']
            available_health = [h for h in health_order if h in df[health_col].unique()]
            if not available_health:
                available_health = sorted(df[health_col].unique())
            input_data[health_col] = st.selectbox("General Health", available_health)
        
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.subheader("Dietary Habits")
        
        # Fried Potato Consumption
        if 'FriedPotato_Consumption' in X.columns:
            input_data['FriedPotato_Consumption'] = st.slider("Fried Potato Consumption (times/week)", 
                                                          int(X['FriedPotato_Consumption'].min()),
                                                          int(X['FriedPotato_Consumption'].max()),
                                                          int(X['FriedPotato_Consumption'].median()))
        
        # Smoking
        smoke_col = find_similar_column(df, ['smoking', 'smoke', 'smoking_history'])
        if smoke_col:
            input_data[smoke_col] = st.selectbox("Smoking History", df[smoke_col].unique())
        
        # Alcohol
        alcohol_col = find_similar_column(df, ['alcohol', 'alcohol_consumption'])
        if alcohol_col:
            input_data[alcohol_col] = st.slider("Alcohol Consumption (days/month)", 
                                             int(df[alcohol_col].min()), 
                                             int(df[alcohol_col].max()), 
                                             int(df[alcohol_col].median()))
        
        # Fruit consumption
        fruit_col = find_similar_column(df, ['fruit', 'fruit_consumption'])
        if fruit_col:
            input_data[fruit_col] = st.slider("Fruit Consumption (times/week)", 
                                          int(df[fruit_col].min()), 
                                          int(df[fruit_col].max()), 
                                          int(df[fruit_col].median()))
        
        # Vegetable consumption
        veg_col = find_similar_column(df, ['vegetable', 'green_vegetables', 'veggies'])
        if veg_col:
            input_data[veg_col] = st.slider("Vegetable Consumption (times/week)", 
                                         int(df[veg_col].min()), 
                                         int(df[veg_col].max()), 
                                         int(df[veg_col].median()))
        
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.subheader("Health Measurements & Activity")
        
        # BMI
        bmi_col = find_similar_column(df, ['bmi', 'body_mass_index'])
        if bmi_col:
            input_data[bmi_col] = st.slider("Body Mass Index (BMI)", 
                                         float(df[bmi_col].min()), 
                                         float(df[bmi_col].max()), 
                                         float(df[bmi_col].median()))
        
        # Physical health days
        phys_health_col = find_similar_column(df, ['physical_health', 'phys_health'])
        if phys_health_col:
            input_data[phys_health_col] = st.slider("Physical Health (days affected in last 30 days)", 
                                                 int(df[phys_health_col].min()), 
                                                 int(df[phys_health_col].max()), 
                                                 int(df[phys_health_col].median()))
        
        # Mental health days
        mental_health_col = find_similar_column(df, ['mental_health', 'ment_health'])
        if mental_health_col:
            input_data[mental_health_col] = st.slider("Mental Health (days affected in last 30 days)", 
                                                   int(df[mental_health_col].min()), 
                                                   int(df[mental_health_col].max()), 
                                                   int(df[mental_health_col].median()))
        
        # Checkup
        if 'Checkup' in X.columns:
            input_data['Checkup'] = st.selectbox("Last Checkup Time", 
                                              X['Checkup'].unique())
        
        # Sleep time
        sleep_col = find_similar_column(df, ['sleep', 'sleep_time'])
        if sleep_col:
            input_data[sleep_col] = st.slider("Sleep Time (hours/night)", 
                                            float(df[sleep_col].min()), 
                                            float(df[sleep_col].max()), 
                                            float(df[sleep_col].median()))
        
        # Physical activity
        activity_col = find_similar_column(df, ['physical_activity', 'exercise'])
        if activity_col:
            input_data[activity_col] = st.selectbox("Engaged in Physical Activity", df[activity_col].unique())
        
        # Walking
        walk_col = find_similar_column(df, ['walking', 'walks'])
        if walk_col:
            input_data[walk_col] = st.selectbox("Engaged in Walking", df[walk_col].unique())
        
        st.markdown('</div>', unsafe_allow_html=True)

    model_choice = st.selectbox("Select prediction model", list(pipelines.keys()))

    if st.button("Predict Heart Disease Risk"):
        # Ensure we only include columns that exist in the original dataframe
        valid_input_data = {k: v for k, v in input_data.items() if k in X.columns}
        input_df = pd.DataFrame([valid_input_data])
        
        # Check if we have all required columns
        missing_cols = set(X.columns) - set(input_df.columns)
        if missing_cols:
            st.markdown('<div class="warning-card">', unsafe_allow_html=True)
            st.warning(f"Note: Some features weren't provided. Prediction may be less accurate. Missing: {', '.join(missing_cols)}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        try:
            # Add missing columns with median values
            for col in missing_cols:
                if col in num_cols:
                    input_df[col] = X[col].median()
                else:
                    # For categorical, use the most frequent value
                    input_df[col] = X[col].mode()[0]
            
            # Reorder columns to match training data
            input_df = input_df[X.columns]
            
            pipeline = pipelines[model_choice]
            prediction = pipeline.predict(input_df)[0]
            prob = pipeline.predict_proba(input_df)[0][1]

            if prediction == 1:
                st.markdown('<div class="prediction-card high-risk">', unsafe_allow_html=True)
                st.subheader(f"High Risk ⚠ ({model_choice})")
                st.markdown(f"The model predicts a high risk of heart disease with a probability of {prob:.2%}.")
                st.markdown("""
                Recommendations:
                - Consult with a healthcare professional
                - Consider lifestyle changes (diet, exercise)
                - Monitor your blood pressure and cholesterol regularly
                - Reduce stress and get adequate sleep
                """)
            else:
                st.markdown('<div class="prediction-card low-risk">', unsafe_allow_html=True)
                st.subheader(f"Low Risk 😊 ({model_choice})")
                st.markdown(f"The model predicts a low risk of heart disease with a probability of {prob:.2%}.")
                st.markdown("""
                Recommendations to maintain heart health:
                - Continue healthy habits
                - Regular check-ups
                - Balanced diet and exercise
                - Maintain healthy weight
                - Avoid smoking and limit alcohol
                """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show probability breakdown with safe progress bar
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("Probability Breakdown")
            st.write(f"Probability of no heart disease: {(1-prob):.2%}")
            st.write(f"Probability of heart disease: {prob:.2%}")
            
            st.markdown('<div class="progress-container">', unsafe_allow_html=True)
            safe_progress_bar(prob)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # LIME explanation for this prediction
            st.subheader("Prediction Explanation")
            
            with st.spinner('Generating prediction explanation...'):
                try:
                    # Get the preprocessor and model from the pipeline
                    preprocessor = pipeline.named_steps['preprocessor']
                    model = pipeline.named_steps['classifier']
                    
                    # Transform the input data
                    input_transformed = preprocessor.transform(input_df)
                    
                    # Get feature names after preprocessing
                    numeric_features = list(num_cols)
                    categorical_features = list(cat_cols)
                    ohe = preprocessor.named_transformers_['cat']
                    ohe_feature_names = ohe.get_feature_names_out(categorical_features)
                    feature_names = numeric_features + list(ohe_feature_names)
                    
                    # Create LIME explainer
                    explainer = LimeTabularExplainer(
                        training_data=preprocessor.transform(X_train),
                        feature_names=feature_names,
                        class_names=['No Heart Disease', 'Heart Disease'],
                        mode='classification',
                        discretize_continuous=True
                    )
                    
                    # Generate LIME explanation
                    exp = explainer.explain_instance(
                        data_row=input_transformed[0],
                        predict_fn=model.predict_proba,
                        num_features=10,
                        top_labels=1
                    )
                    
                    st.markdown('<div class="lime-plot">', unsafe_allow_html=True)
                    
                    # Display LIME explanation
                    st.write("How each feature contributed to this prediction:")
                    
                    # Get the explanation as HTML and display it
                    html = exp.as_html()
                    st.components.v1.html(html, height=800, scrolling=True)
                    
                    st.markdown("""
                    How to read this explanation:
                    - The left side shows the features that most influenced the prediction
                    - Orange/red bars indicate features that increased the risk prediction
                    - Blue bars indicate features that decreased the risk prediction
                    - The values show the actual feature values for this case
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error generating LIME explanation: {str(e)}")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.error("Please check if all required fields are filled correctly.")
            
    # Debug information 
    if st.sidebar.checkbox("Show debug info"):
        st.write("### Debug Information")
        st.write("Input data:", input_data)
        st.write("Processed input:", input_df)
        st.write("Missing columns:", missing_cols)
        st.write("Full error details:", repr(e))