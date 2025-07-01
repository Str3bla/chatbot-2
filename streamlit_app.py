import streamlit as st
import requests
import pandas as pd

# =============================================================================
# WORKING ZOHO CREDENTIALS (from successful curl test)
# =============================================================================
ZOHO_ACCESS_TOKEN = "1000.ea224242bd9685cde0dc047bd9374d3c.f10a0ddfe7e39a1de770c768b716b89d"
ZOHO_BASE_URL = "https://recruit.zoho.com/recruit/v2"
JOB_ID = "ZR_1_JOB"

# =============================================================================
# SIMPLE API TEST
# =============================================================================

def fetch_job_data():
    """Fetch job data from Zoho - exact same as working curl command"""
    url = f"{ZOHO_BASE_URL}/JobOpenings/{JOB_ID}"
    headers = {
        'Authorization': f'Zoho-oauthtoken {ZOHO_ACCESS_TOKEN}',
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
        return None

# =============================================================================
# STREAMLIT APP
# =============================================================================

st.title("🎯 Zoho API Test - Proof of Concept")
st.write("Testing the exact same API call that worked in curl")

st.markdown("---")
st.subheader("📋 API Details")
st.write(f"**URL:** {ZOHO_BASE_URL}/JobOpenings/{JOB_ID}")
st.write(f"**Job ID:** {JOB_ID}")
st.write(f"**Access Token:** {ZOHO_ACCESS_TOKEN[:20]}...")

st.markdown("---")

if st.button("🚀 Test API Call", type="primary"):
    with st.spinner("Calling Zoho API..."):
        data = fetch_job_data()
        
        if data:
            st.success("✅ API call successful!")
            
            # Show raw response structure
            st.subheader("📊 Raw API Response")
            st.json(data)
            
            # Extract job data if available
            if 'data' in data and len(data['data']) > 0:
                job_info = data['data'][0]
                
                st.subheader("🎯 Job Information")
                
                # Create simple table of key fields
                job_fields = []
                for key, value in job_info.items():
                    if isinstance(value, (str, int, float)) and value:  # Only simple fields
                        job_fields.append({"Field": key, "Value": str(value)[:100]})  # Truncate long values
                
                if job_fields:
                    df = pd.DataFrame(job_fields)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Show job description if it exists
                job_description = job_info.get('Job_Description', '')
                if job_description:
                    st.subheader("📄 Job Description")
                    st.text_area("", job_description[:500] + "..." if len(job_description) > 500 else job_description, height=200, disabled=True)
            
        else:
            st.error("❌ API call failed")

st.markdown("---")
st.info("💡 This is the minimal test to prove Zoho API integration works. Once this succeeds, we can build the similarity engine on top of it.")
