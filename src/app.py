import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
import os

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="LoanGuard",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-size: 1rem;
        font-weight: bold;
        border: none;
    }
    .risk-low {
        background-color: #dcfce7;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #16a34a;
    }
    .risk-medium {
        background-color: #fef9c3;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #ca8a04;
    }
    .risk-high {
        background-color: #fee2e2;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #dc2626;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("# 🛡️ LoanGuard")
st.markdown("### AI-Powered Loan Default Risk Assessment")
st.divider()

# Sidebar
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.info("""
        LoanGuard uses an XGBoost model trained on 300,000+ loan applications 
        to predict default risk in real time.
        
        **Model Performance**
        - ROC-AUC: 0.75
        - Auto drift detection
        - Auto retraining pipeline
    """)
    
    st.markdown("## 🔗 API Status")
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        if response.status_code == 200:
            st.success("API is online ✅")
        else:
            st.error("API is offline ❌")
    except:
        st.error("API is offline ❌")

# Main content — two columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📋 Applicant Information")
    
    with st.form("loan_form"):
        st.markdown("**Financial Details**")
        
        income = st.number_input(
            "Annual Income (₹)", 
            min_value=10000, 
            max_value=10000000, 
            value=500000,
            step=10000
        )
        
        credit_amount = st.number_input(
            "Loan Amount Requested (₹)", 
            min_value=10000, 
            max_value=5000000, 
            value=1000000,
            step=10000
        )
        
        annuity = st.number_input(
            "Monthly Annuity (₹)", 
            min_value=1000, 
            max_value=500000, 
            value=25000,
            step=1000
        )
        
        goods_price = st.number_input(
            "Goods Price (₹)", 
            min_value=10000, 
            max_value=5000000, 
            value=800000,
            step=10000
        )
        
        st.markdown("**Personal Details**")
        
        age = st.slider("Age", min_value=18, max_value=70, value=35)
        employment_years = st.slider("Years Employed", min_value=0, max_value=40, value=5)
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        family_members = st.number_input("Family Members", min_value=1, max_value=15, value=2)
        
        st.markdown("**Other Details**")
        
        gender = st.selectbox("Gender", ["Female", "Male"])
        owns_car = st.selectbox("Owns a Car?", ["No", "Yes"])
        owns_realty = st.selectbox("Owns Property?", ["No", "Yes"])
        contract_type = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
        
        ext1 = st.slider("External Score 1", 0.0, 1.0, 0.6)
        ext2 = st.slider("External Score 2", 0.0, 1.0, 0.6)
        ext3 = st.slider("External Score 3", 0.0, 1.0, 0.6)
        
        submitted = st.form_submit_button("🔍 Assess Risk", use_container_width=True)

with col2:
    st.markdown("### 📊 Risk Assessment Result")
    
    if submitted:
        payload = {
            "AMT_INCOME_TOTAL": float(income),
            "AMT_CREDIT": float(credit_amount),
            "AMT_ANNUITY": float(annuity),
            "AMT_GOODS_PRICE": float(goods_price),
            "DAYS_BIRTH": -(age * 365),
            "DAYS_EMPLOYED": -(employment_years * 365),
            "DAYS_REGISTRATION": -3648.0,
            "DAYS_ID_PUBLISH": -2700,
            "CNT_FAM_MEMBERS": float(family_members),
            "CNT_CHILDREN": int(children),
            "EXT_SOURCE_1": float(ext1),
            "EXT_SOURCE_2": float(ext2),
            "EXT_SOURCE_3": float(ext3),
            "CODE_GENDER": 1 if gender == "Male" else 0,
            "NAME_CONTRACT_TYPE": 0 if contract_type == "Cash loans" else 1,
            "FLAG_OWN_CAR": 1 if owns_car == "Yes" else 0,
            "FLAG_OWN_REALTY": 1 if owns_realty == "Yes" else 0
        }
        
        with st.spinner("Analyzing application..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=10
                )
                result = response.json()
                
                risk = result["risk_level"]
                prob = result["default_probability"]
                recommendation = result["recommendation"]
                
                # Risk meter
                st.markdown("#### Default Probability")
                st.progress(prob)
                st.markdown(f"**{prob * 100:.1f}%** chance of default")
                
                st.divider()
                
                # Risk badge
                if risk == "LOW":
                    st.markdown(f"""
                        <div class="risk-low">
                            <h3>🟢 LOW RISK</h3>
                            <p>{recommendation}</p>
                        </div>
                    """, unsafe_allow_html=True)
                elif risk == "MEDIUM":
                    st.markdown(f"""
                        <div class="risk-medium">
                            <h3>🟡 MEDIUM RISK</h3>
                            <p>{recommendation}</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="risk-high">
                            <h3>🔴 HIGH RISK</h3>
                            <p>{recommendation}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.divider()
                
                # Application summary
                st.markdown("#### 📋 Application Summary")
                summary_data = {
                    "Field": ["Annual Income", "Loan Amount", "Monthly Annuity", 
                             "Credit/Income Ratio", "Age", "Employment Years"],
                    "Value": [
                        f"₹{income:,.0f}",
                        f"₹{credit_amount:,.0f}",
                        f"₹{annuity:,.0f}",
                        f"{credit_amount/income:.2f}x",
                        f"{age} years",
                        f"{employment_years} years"
                    ]
                }
                st.table(pd.DataFrame(summary_data))
                
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")
                st.info("Make sure the FastAPI server is running on port 8000")
    
    else:
        st.info("👈 Fill in the applicant details and click **Assess Risk** to get a prediction.")
        
        st.markdown("#### 💡 How it works")
        st.markdown("""
        1. **Enter** applicant financial and personal details
        2. **Click** Assess Risk to call the ML model
        3. **Review** the risk score and recommendation
        4. **Monitor** — all predictions are logged for drift detection
        """)

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model", "XGBoost")
with col2:
    st.metric("ROC-AUC", "0.75")
with col3:
    st.metric("Training Samples", "246,008")