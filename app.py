# =================================================================
# Streamlit Credit Risk Assessment GUI - V4 (Corrected)
# =================================================================
# This script builds the final, simplified version of the tool.

# --- 1. SETUP ---
# pip install streamlit pillow
import streamlit as st
import datetime
import math
from PIL import Image # Used to display images

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Credit Risk Assessment Tool",
    layout="wide"
)

# --- 3. MODEL COEFFICIENTS (Corrected with Final SMOTE values) ---
COEFFICIENTS = {
    'intercept': 0.1118,
    'loan_term': 0.01796,
    'loan_amount': 0.00009528,
    'age': -0.008217,
    'city_prague': 0.7526,
    'city_unknown': -0.1912,
    'edu_secondary': -1.657,
    'edu_university': -2.580
}

# --- 4. HELPER FUNCTIONS ---

def calculate_pd(age, loan_amount, loan_term, city, education):
    """Calculates the probability of default based on user inputs."""
    # Handle categorical variables to get their coefficient values
    if city == "Prague":
        city_coeff = COEFFICIENTS['city_prague']
    elif city == "Unknown":
        city_coeff = COEFFICIENTS['city_unknown']
    else: # Brno is the baseline
        city_coeff = 0

    if education == "Secondary":
        edu_coeff = COEFFICIENTS['edu_secondary']
    elif education == "University":
        edu_coeff = COEFFICIENTS['edu_university']
    else: # Primary is the baseline
        edu_coeff = 0
        
    log_odds = (COEFFICIENTS['intercept'] + (loan_term * COEFFICIENTS['loan_term']) +
                (loan_amount * COEFFICIENTS['loan_amount']) + (age * COEFFICIENTS['age']) +
                city_coeff + edu_coeff)
    probability = 1 / (1 + math.exp(-log_odds))
    return probability

def get_applicant_profile(age, education, loan_amount, city):
    """Matches an applicant to a new, more robust set of pre-defined profiles."""
    if loan_amount > 20000:
        profile_name = "High-Value Applicant"
        profile_details = "- Loan Amount > €20,000"
        profile_insight = "This is a significant loan amount where defaults are costly. A manual review by a senior officer is always recommended for this segment, regardless of the PD score."
        return profile_name, profile_details, profile_insight
        
    elif education == "Primary" and city == "Prague":
        profile_name = "High-Risk Urban Applicant"
        profile_details = "- Education is Primary\n- City is Prague"
        profile_insight = "This profile combines two factors associated with higher risk. Historically, this group has a higher default rate. Scrutinize repayment capacity."
        return profile_name, profile_details, profile_insight

    elif education == "University" and age > 30:
        profile_name = "Established Professional"
        profile_details = "- Education is University\n- Age is over 30"
        profile_insight = "This profile has historically shown a very low default rate. They are generally our safest customers."
        return profile_name, profile_details, profile_insight

    elif age < 28:
        profile_name = "Young & Aspiring Applicant"
        profile_details = "- Age is under 28"
        profile_insight = "For this younger demographic, stability of income is the most critical factor to verify. Their risk profile can vary greatly."
        return profile_name, profile_details, profile_insight
        
    else:
        profile_name = "Standard Applicant"
        profile_details = "N/A"
        profile_insight = "This applicant does not match a specific high-risk or low-risk profile. The model's PD score should be the primary guide for the lending decision."
        return profile_name, profile_details, profile_insight


# --- 5. UI LAYOUT (Simplified - No Tabs) ---
st.title("Credit Risk Decision Support Tool")
st.write("A tool for loan officers to assess applicant risk using our predictive model.")
st.divider()

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("Applicant Data")
    st.write("Please enter the applicant's details. All fields are mandatory.")

    with st.form(key='applicant_form'):
        year_of_application = st.number_input("Year of Application", min_value=2000, max_value=datetime.date.today().year, value=datetime.date.today().year, step=1)
        year_of_birth = st.number_input("Year of Birth", min_value=1940, max_value=datetime.date.today().year, value=1990, step=1)
        loan_amount = st.number_input("Loan Amount (€)", min_value=500, value=10000, step=500)
        loan_term = st.slider("Loan Term (Months)", min_value=6, max_value=72, value=36, step=3)
        city = st.selectbox("City", options=["", "Prague", "Brno"], help="Select the applicant's city.")
        education = st.selectbox("Education Level", options=["", "Primary", "Secondary", "University"], help="Select the applicant's highest education level.")
        submit_button = st.form_submit_button(label='Assess Risk')

with col2:
    st.subheader("Assessment Results")

    if submit_button:
        if not all([city, education]):
            st.error("Error: All fields are mandatory. Please select a City and Education level.")
        elif year_of_birth >= year_of_application:
            st.error("Error: Year of Birth must be before Year of Application.")
        else:
            age = year_of_application - year_of_birth
            pd_probability = calculate_pd(age, loan_amount, loan_term, city, education)

            if loan_amount > 15000:
                if 0.3 <= pd_probability <= 0.7: assessment = "Unsure Default (High-Value)"
                elif pd_probability > 0.7: assessment = "Sure Default"
                else: assessment = "Sure Non-Default"
            else:
                if 0.4 <= pd_probability <= 0.6: assessment = "Unsure Default"
                elif pd_probability > 0.6: assessment = "Sure Default"
                else: assessment = "Sure Non-Default"

            st.metric(label="Probability of Default", value=f"{pd_probability:.2%}")

            if "Unsure" in assessment:
                st.warning(f"**Final Assessment:** {assessment}")
                st.info("💡 **Suggestion:** This applicant is in a 'grey area'. A manual review by a senior loan officer is recommended. Consider approving a partial loan amount to mitigate risk.")
            elif "Sure Default" in assessment:
                st.error(f"**Final Assessment:** {assessment}")
                st.info("💡 **Suggestion:** This applicant has a high risk of default. Declining the loan is recommended based on the model's prediction.")
            else:
                st.success(f"**Final Assessment:** {assessment}")
                st.info("💡 **Suggestion:** This applicant has a low risk of default. Approving the loan is recommended.")
            
            profile_name, profile_details, profile_insight = get_applicant_profile(age, education, loan_amount, city)
            with st.expander("View Applicant Profile Analysis"):
                st.write(f"**Matched Profile:** {profile_name}")
                if profile_details != "N/A":
                    st.write(f"**Defining Characteristics:**")
                    st.markdown(profile_details)
                st.write(f"**Historical Context:** {profile_insight}")
                st.write(f"This provides a benchmark to compare the model's specific prediction against the historical average for this type of customer.")
