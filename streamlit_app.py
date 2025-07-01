import openai
import numpy as np
import pandas as pd
import streamlit as st
import requests
import json
from typing import List, Optional
import re

# =============================================================================
# ZOHO RECRUIT API CREDENTIALS - WORKING CREDENTIALS
# =============================================================================
ZOHO_CLIENT_ID = "1000.AL6419HX3THLR4U73ZGWVMPGPTITNX"
ZOHO_CLIENT_SECRET = "361cf827b2cb1e0903843319f0eb7856e3d6fdaac6" 
ZOHO_REFRESH_TOKEN = "1000.7e3393a940e040d2fd0da57655803e28f.1bb619084ba92e2dba1098cdab304a15"
ZOHO_BASE_URL = "https://recruit.zoho.com/recruit/v2"
TARGET_JOB_OPENING_ID = "821313000005285257"  # Internal record ID for "Analyst II - ATS Engineering"

# =============================================================================
# ZOHO API FUNCTIONS
# =============================================================================

def get_zoho_access_token() -> Optional[str]:
    """Get fresh access token using refresh token"""
    token_url = "https://accounts.zoho.com/oauth/v2/token"
    
    payload = {
        'grant_type': 'refresh_token',
        'client_id': ZOHO_CLIENT_ID,
        'client_secret': ZOHO_CLIENT_SECRET,
        'refresh_token': ZOHO_REFRESH_TOKEN
    }
    
    try:
        response = requests.post(token_url, data=payload)
        if response.status_code == 200:
            token_data = response.json()
            return token_data.get('access_token')
        else:
            st.error(f"❌ Failed to get access token: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"❌ Error getting access token: {str(e)}")
        return None

def fetch_job_from_zoho(job_id: str) -> Optional[str]:
    """Fetch job description from Zoho Recruit"""
    access_token = get_zoho_access_token()
    if not access_token:
        return None
    
    url = f"{ZOHO_BASE_URL}/JobOpenings/{job_id}"
    headers = {
        'Authorization': f'Zoho-oauthtoken {access_token}',
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and len(data['data']) > 0:
                job_data = data['data'][0]
                job_description = job_data.get('Job_Description', '')
                job_title = job_data.get('Job_Opening_Name', 'Unknown Job')
                
                # Clean HTML tags from description
                clean_description = re.sub(r'<[^>]+>', '', job_description)
                
                return {
                    'title': job_title,
                    'description': clean_description.strip(),
                    'raw_data': job_data
                }
        
        st.error(f"❌ Job {job_id} not found or API error: {response.status_code}")
        return None
        
    except Exception as e:
        st.error(f"❌ Error fetching job: {str(e)}")
        return None

# =============================================================================
# SIMILARITY ENGINE
# =============================================================================

def get_embedding(text: str, api_key: str) -> Optional[List[float]]:
    """Get OpenAI embedding for text"""
    try:
        openai.api_key = api_key
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"❌ Error getting embedding: {str(e)}")
        return None

def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    
    return dot_product / (norm_v1 * norm_v2)

def get_similarity_analysis(score: float) -> str:
    """Convert similarity score to qualitative analysis"""
    if score >= 0.95:
        return "Exact Match"
    elif score >= 0.85:
        return "Very High Match"
    elif score >= 0.70:
        return "High Match"
    elif score >= 0.50:
        return "Moderate Match"
    elif score >= 0.30:
        return "Low Match"
    else:
        return "Very Low Match"

# =============================================================================
# HARDCODED COMPARISON JOBS
# =============================================================================

COMPARISON_JOBS = {
    "Backend Engineer": {
        "description": """
        Senior Software Engineer - Backend Development
        
        We are seeking a Senior Software Engineer to join our backend development team. 
        The ideal candidate will have 5+ years of experience in Python, Django, and PostgreSQL.
        
        Responsibilities:
        - Design and implement scalable backend systems
        - Work with REST APIs and microservices architecture
        - Collaborate with frontend developers and product managers
        - Write clean, maintainable code with comprehensive tests
        
        Requirements:
        - Bachelor's degree in Computer Science or related field
        - Strong experience with Python and Django framework
        - Knowledge of database design and optimization
        - Experience with cloud platforms (AWS, GCP)
        - Excellent problem-solving and communication skills
        """,
        "time_to_fill": 76
    },
    
    "Marketing Manager": {
        "description": """
        Marketing Manager - Digital Campaigns
        
        We're looking for a creative Marketing Manager to lead our digital marketing initiatives.
        This role requires 3+ years of marketing experience with a focus on digital channels.
        
        What you'll do:
        - Develop and execute digital marketing campaigns
        - Manage social media presence and content strategy
        - Analyze campaign performance and ROI
        - Collaborate with design and content teams
        - Manage marketing budget and vendor relationships
        
        What we're looking for:
        - Bachelor's degree in Marketing, Communications, or related field
        - Proven experience with Google Ads, Facebook Ads, and SEO
        - Strong analytical skills and data-driven mindset
        - Experience with marketing automation tools
        - Creative thinking and project management skills
        """,
        "time_to_fill": 45
    }
}

# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    st.title("🎯 Sourcingology: Job Similarity Engine")
    st.subheader("Compare Zoho job against historical data")
    
    # API Key input
    api_key = st.text_input("OpenAI API Key", type="password", 
                           help="Enter your OpenAI API key for embeddings")
    
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key to continue")
        return
    
    # Show target job info
    st.markdown("---")
    st.subheader("🎯 Target Job from Zoho")
    st.info(f"**Job Opening ID:** {TARGET_JOB_OPENING_ID} (Analyst II - ATS Engineering)")
    st.caption("Using internal record ID from Zoho Recruit API")
    
    # Show comparison jobs
    st.subheader("📊 Historical Jobs for Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔧 Backend Engineer**")
        st.write(f"Time to Fill: {COMPARISON_JOBS['Backend Engineer']['time_to_fill']} days")
        with st.expander("View Job Description"):
            st.text_area("", COMPARISON_JOBS['Backend Engineer']['description'], height=200, disabled=True)
    
    with col2:
        st.markdown("**📈 Marketing Manager**")
        st.write(f"Time to Fill: {COMPARISON_JOBS['Marketing Manager']['time_to_fill']} days")
        with st.expander("View Job Description"):
            st.text_area("", COMPARISON_JOBS['Marketing Manager']['description'], height=200, disabled=True)
    
    # Main comparison button
    st.markdown("---")
    if st.button("🚀 Run Similarity Analysis", type="primary"):
        
        with st.spinner("🔄 Fetching job from Zoho..."):
            # Step 1: Fetch job from Zoho
            zoho_job = fetch_job_from_zoho(TARGET_JOB_OPENING_ID)
            
            if not zoho_job:
                st.error("❌ Failed to fetch job from Zoho. Check your credentials and job ID.")
                return
            
            st.success(f"✅ Successfully fetched: {zoho_job['title']}")
            
            # Show fetched job description
            with st.expander("📄 Job Description from Zoho", expanded=True):
                st.write(f"**Title:** {zoho_job['title']}")
                st.text_area("Description", zoho_job['description'], height=200, disabled=True)
        
        with st.spinner("🧠 Generating embeddings and calculating similarities..."):
            # Step 2: Generate embeddings
            zoho_embedding = get_embedding(zoho_job['description'], api_key)
            if not zoho_embedding:
                st.error("❌ Failed to generate embedding for Zoho job")
                return
            
            # Step 3: Compare against each historical job
            results = []
            
            for job_name, job_data in COMPARISON_JOBS.items():
                # Get embedding for comparison job
                comp_embedding = get_embedding(job_data['description'], api_key)
                if not comp_embedding:
                    st.error(f"❌ Failed to generate embedding for {job_name}")
                    continue
                
                # Calculate similarity
                similarity = calculate_cosine_similarity(zoho_embedding, comp_embedding)
                analysis = get_similarity_analysis(similarity)
                
                results.append({
                    'Zoho Job Opening ID': TARGET_JOB_OPENING_ID,
                    'Compared To': job_name,
                    'Similarity Score': round(similarity, 4),
                    'Analysis': analysis,
                    'Time to Fill': f"{job_data['time_to_fill']} days"
                })
            
            # Step 4: Display results
            if results:
                st.markdown("---")
                st.subheader("📊 Similarity Results")
                
                # Create and display results table
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Key insights
                st.markdown("---")
                st.subheader("💡 Key Insights")
                
                # Find best match
                best_match = max(results, key=lambda x: x['Similarity Score'])
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Best Match", best_match['Compared To'])
                
                with col2:
                    st.metric("Similarity Score", f"{best_match['Similarity Score']:.3f}")
                
                with col3:
                    st.metric("Predicted Time to Fill", best_match['Time to Fill'])
                
                # Recommendations
                st.markdown("---")
                st.subheader("🎯 Sourcing Recommendations")
                
                if best_match['Similarity Score'] >= 0.7:
                    st.success(f"""
                    **High similarity detected!** 
                    
                    Based on the {best_match['Analysis'].lower()} with **{best_match['Compared To']}**, 
                    we recommend:
                    
                    - **Expected time to fill:** {best_match['Time to Fill']}
                    - **Sourcing approach:** Use similar strategies as {best_match['Compared To']}
                    - **Confidence level:** High (similarity: {best_match['Similarity Score']:.3f})
                    """)
                else:
                    st.warning(f"""
                    **Moderate similarity detected.**
                    
                    The closest match is **{best_match['Compared To']}** ({best_match['Similarity Score']:.3f} similarity),
                    but this is a relatively unique role. Consider:
                    
                    - **Custom sourcing strategy** may be needed
                    - **Time to fill:** Potentially longer than {best_match['Time to Fill']}
                    - **Market research:** Investigate similar roles in the market
                    """)
            
            st.markdown("---")
            st.subheader("🧠 How This Works")
            st.markdown("""
            **Vector Similarity Analysis:**
            1. **Job descriptions** are converted to high-dimensional vectors using OpenAI embeddings
            2. **Cosine similarity** measures the angle between vectors (semantic similarity)
            3. **Historical data** provides time-to-fill predictions based on similar roles
            4. **Sourcing strategies** can be adapted from similar successful hires
            """)

if __name__ == "__main__":
    main()
