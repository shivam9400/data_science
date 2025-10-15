import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Complaint Outcome Predictor", layout="wide")

@st.cache_resource
def load_models():
    """Loads the trained model pipelines."""
    try:
        model_financial = joblib.load('artifacts/logistic_is_financial_relief.joblib')
        model_upheld = joblib.load('artifacts/logistic_is_upheld.joblib')
        return model_upheld, model_financial
    except FileNotFoundError:
        st.error("Model files not found. Ensure 'logistic_is_financial_relief.joblib' and 'logistic_is_upheld.joblib' are in the root directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()
    
# Load models immediately
MODEL_UPHELD, MODEL_FINANCIAL = load_models()

def run_inference(data_dict: dict) -> tuple:
    """
    Takes a dictionary of raw consumer complaint features and returns
    the predicted probabilities.
    """
    # 1. Create DataFrame: Ensure all 9 required features are present
    required_features = [
        'Product', 'Sub-product', 'Issue', 'Sub-issue', 'Company', 
        'State', 'Submitted via', 'Consumer complaint narrative'
    ]
    
    # Create a DataFrame from the input dictionary
    input_df = pd.DataFrame([data_dict])[required_features]
    
    # Crucial: Fill any potential NaNs in the string columns with empty strings 
    # to match the preprocessing done during training
    input_df = input_df.fillna('')

    # 2. Predict P(Upheld) - P1
    # The pipeline handles all complex preprocessing
    p_upheld = MODEL_UPHELD.predict_proba(input_df)[:, 1][0]
    
    # 3. Predict P(Financial Relief) - P2
    p_financial = MODEL_FINANCIAL.predict_proba(input_df)[:, 1][0]
    
    # 4. Calculate Overall Probability (Combined Logic: P_overall = P1 * P2)
    p_overall_financial_relief = p_upheld * p_financial 

    return p_upheld, p_financial, p_overall_financial_relief


# Streamlit App Interface
st.title("Complaint Outcome Predictor")
st.markdown("Enter your complaint details to estimate your chance of a successful outcome and financial relief.")

# --- Input Options (Customize these based on your top categories from EDA) ---
PRODUCT_OPTIONS = ['Mortgage', 'Credit reporting', 'Debt collection', 'Credit card', 'Bank account or service', 'Other']
ISSUE_OPTIONS = ['Incorrect information on credit report', 'Loan modification,collection,foreclosure', 'Managing the loan or lease', 'Communication tactics', 'Other']
STATE_OPTIONS = ['CA', 'TX', 'FL', 'NY', 'MI', 'PA', 'Other']
SUBMITTED_OPTIONS = ['Web', 'Referral', 'Phone', 'Other']


with st.form("complaint_form"):
    st.subheader("Complaint Details")

    col1, col2 = st.columns(2)
    with col1:
        product = st.selectbox("1. Product:", options=PRODUCT_OPTIONS, index=0)
        sub_product = st.text_input("2. Sub-product:", value="", placeholder="e.g., Vehicle loan, FHA mortgage")
        issue = st.selectbox("3. Issue:", options=ISSUE_OPTIONS, index=0)
        sub_issue = st.text_input("4. Sub-issue:", value="", placeholder="e.g., Account status, Frequent or repeated calls")
        
    with col2:
        company = st.text_input("5. Company Name:", value="WELLS FARGO & COMPANY", placeholder="e.g., WELLS FARGO & COMPANY")
        state = st.selectbox("6. State:", options=STATE_OPTIONS, index=0)
        submitted_via = st.selectbox("7. Submitted Via:", options=SUBMITTED_OPTIONS, index=0)

    st.subheader("Narrative")
    # A. Text Input (The most important feature)
    default_narrative = "I have an incorrect account status on my credit report that I disputed 30 days ago. The company failed to remove the old information."
    narrative = st.text_area("8. Consumer Complaint Narrative:", 
                             value=default_narrative,
                             height=200)

    submitted = st.form_submit_button("Predict Outcome")

# Display Results
if submitted:
    if not narrative or not company:
        st.error("Please enter the complaint narrative and the company name.")
    else:
        # Create input dictionary
        input_data = {
            'Product': product,
            'Sub-product': sub_product, # NEW
            'Issue': issue,
            'Sub-issue': sub_issue,     # NEW
            'Company': company,
            'State': state,
            'Submitted via': submitted_via,
            'Consumer complaint narrative': narrative,
        }
        
        with st.spinner('Calculating probabilities...'):
            p_upheld, p_financial, p_overall = run_inference(input_data)
        
        st.success("Prediction Complete!")
        
        # Display the results clearly to the consumer
        col_upheld, col_financial, col_overall = st.columns(3)
        
        with col_upheld:
             st.header("1. Prob (Upheld)")
             st.metric(label="Likelihood of Any Relief", value=f"{p_upheld*100:.1f}%")

        with col_financial:
            st.header("2. Prob (Financial Relief)")
            st.metric(label="Likelihood of Monetary Relief", value=f"{p_financial*100:.1f}%")

        with col_overall:
            st.header("3. Overall Prob (Financial Relief)")
            st.metric(label="Combined Final Chance (P1 * P2)", value=f"{p_overall*100:.1f}%")

        st.markdown(f"---")
        st.markdown(f"### **Conclusion:**")
        st.info(f"Based on your inputs, your overall chance of receiving **financial relief** from the complaint process is **{p_overall*100:.1f}%**.")