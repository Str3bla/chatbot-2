import openai
import numpy as np
import pandas as pd
import streamlit as st
import requests
import json
from typing import List, Dict, Tuple, Optional
import os

# =============================================================================
# ZOHO RECRUIT API CREDENTIALS - EDIT THESE VALUES
# =============================================================================
# Self Client Credentials (Recommended for Free Tier)
ZOHO_CLIENT_ID = "1000.AL6419HX3THLR4U73ZGWVMPGPTITNX"
ZOHO_CLIENT_SECRET = "361cf827b2cb1e0903843319f0eb7856e3d6fdaac6" 
ZOHO_REFRESH_TOKEN = "your_refresh_token_here"  # You'll get this from the self-client flow

ZOHO_BASE_URL = "https://recruit.zoho.com/recruit/v2"
TARGET_JOB_OPENING_ID = "ZR_1_JOB"

# Data center options - change if needed
ZOHO_ACCOUNTS_URL = "https://accounts.zoho.com"  # US
# ZOHO_ACCOUNTS_URL = "https://accounts.zoho.eu"    # Europe
# ZOHO_ACCOUNTS_URL = "https://accounts.zoho.in"    # India  
# ZOHO_ACCOUNTS_URL = "https://accounts.zoho.com.au" # Australia

# =============================================================================
# ZOHO API HELPER FUNCTIONS - NEW SECTION
# =============================================================================

def get_zoho_access_token(client_id: str, client_secret: str, refresh_token: str) -> Optional[str]:
    """
    Exchange refresh token for access token with Zoho Recruit API
    
    Args:
        client_id: Zoho Client ID
        client_secret: Zoho Client Secret  
        refresh_token: Zoho Refresh Token
        
    Returns:
        Access token string or None if failed
    """
    token_url = f"{ZOHO_ACCOUNTS_URL}/oauth/v2/token"
    
    # Validate inputs first
    validation_errors = []
    
    if not client_id or client_id == "your_client_id_here":
        validation_errors.append("❌ Client ID not set or still placeholder")
    elif len(client_id) < 20:
        validation_errors.append("❌ Client ID seems too short (should be ~65 characters)")
        
    if not client_secret or client_secret == "your_client_secret_here":
        validation_errors.append("❌ Client Secret not set or still placeholder")
    elif len(client_secret) < 30:
        validation_errors.append("❌ Client Secret seems too short (should be ~32+ characters)")
        
    if not refresh_token or refresh_token == "your_refresh_token_here":
        validation_errors.append("❌ Refresh Token not set or still placeholder")
    elif refresh_token.startswith(('http://', 'https://')):
        validation_errors.append("❌ Refresh Token appears to be a URL - it should be an alphanumeric string")
    elif len(refresh_token) < 50:
        validation_errors.append("❌ Refresh Token seems too short (should be 100+ characters)")
    
    if validation_errors:
        st.error("**Validation Errors:**")
        for error in validation_errors:
            st.write(error)
        st.markdown("""
        **What you need:**
        - **Client ID**: Long string like `1000.XXXXXXXXXX...` (~65 chars)
        - **Client Secret**: String like `abc123def456...` (~32+ chars)  
        - **Refresh Token**: Long alphanumeric string (~100+ chars), NOT a URL
        
        **How to get a proper refresh token:**
        1. Go to [Zoho Developer Console](https://api-console.zoho.com/)
        2. Create/select your app
        3. Go to Client Secret tab
        4. Generate a refresh token using the "Generate Code" process
        5. The refresh token should look like: `1000.abc123def456...` (long string)
        """)
        return None
    
    payload = {
        'refresh_token': refresh_token,
        'client_id': client_id,
        'client_secret': client_secret,
        'grant_type': 'refresh_token'
    }
    
    try:
        # Debug: Show what we're sending (without sensitive data)
        st.write("🔍 **Debug Info:**")
        st.write(f"- Token URL: {token_url}")
        st.write(f"- Client ID: {client_id[:15]}..." if len(client_id) > 15 else f"- Client ID: {client_id}")
        st.write(f"- Client Secret: {client_secret[:10]}..." if len(client_secret) > 10 else f"- Client Secret: {client_secret}")
        st.write(f"- Refresh Token: {refresh_token[:20]}..." if len(refresh_token) > 20 else f"- Refresh Token: {refresh_token}")
        
        response = requests.post(token_url, data=payload)
        
        # Debug: Show response details
        st.write(f"- Response Status: {response.status_code}")
        
        if response.status_code != 200:
            # Try to extract meaningful error from HTML response
            error_text = response.text
            if "An error occurred" in error_text:
                st.error(f"❌ Zoho returned an error page (HTTP {response.status_code})")
                st.error("This usually means invalid credentials or wrong data center URL")
            else:
                st.error(f"❌ HTTP {response.status_code}: {response.text[:500]}...")
            return None
        
        token_data = response.json()
        
        # Debug: Show response structure (without sensitive token)
        st.write(f"- Response Keys: {list(token_data.keys())}")
        
        if 'access_token' in token_data:
            st.success("✅ Successfully got access token!")
            return token_data.get('access_token')
        else:
            st.error(f"❌ No access_token in response: {token_data}")
            return None
        
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Request failed: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"❌ Invalid JSON response: {str(e)}")
        st.error("This usually means you got an HTML error page instead of JSON")
        return None

def fetch_job_description_from_zoho(job_opening_id: str, access_token: str) -> Optional[str]:
    """
    Fetch job description from Zoho Recruit API
    
    Args:
        job_opening_id: The Job Opening ID to fetch (e.g., "ZR_1_JOB")
        access_token: Valid Zoho access token
        
    Returns:
        Job description text or None if failed
    """
    url = f"{ZOHO_BASE_URL}/JobOpenings/{job_opening_id}"
    
    headers = {
        'Authorization': f'Zoho-oauthtoken {access_token}',
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract job description from the response
        if 'data' in data and len(data['data']) > 0:
            job_data = data['data'][0]
            job_description = job_data.get('Job_Description', '')
            
            if job_description:
                # Clean up rich text formatting if needed
                # Remove HTML tags for better embedding processing
                import re
                clean_description = re.sub(r'<[^>]+>', '', job_description)
                return clean_description.strip()
            else:
                st.error(f"No job description found for Job Opening ID: {job_opening_id}")
                return None
        else:
            st.error(f"No data found for Job Opening ID: {job_opening_id}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch job from Zoho: {str(e)}")
        return None
    except json.JSONDecodeError:
        st.error("Invalid response format from Zoho API")
        return None
    except Exception as e:
        st.error(f"Unexpected error fetching job: {str(e)}")
        return None

# =============================================================================
# JOB SIMILARITY ENGINE - ORIGINAL CODE (MOSTLY UNCHANGED)
# =============================================================================

class JobSimilarityEngine:
    def __init__(self, api_key: str = None):
        """
        Initialize the Job Similarity Engine
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment
        """
        if api_key:
            openai.api_key = api_key
        else:
            openai.api_key = os.getenv('OPENAI_API_KEY')
        
        self.model = "text-embedding-3-small"
        self.job_data = []
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get vector embedding for a text using OpenAI's embedding model
        
        Args:
            text: The job description text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            response = openai.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            st.error(f"Error getting embedding: {str(e)}")
            return None
    
    def add_job_description(self, req_id: str, description: str, time_to_fill: int = 89):
        """
        Add a job description to the engine and generate its embedding
        
        Args:
            req_id: Requisition ID
            description: Job description text
            time_to_fill: Time to fill in days
        """
        embedding = self.get_embedding(description)
        if embedding:
            self.job_data.append({
                'req_id': req_id,
                'description': description,
                'embedding': embedding,
                'time_to_fill': time_to_fill
            })
            return True
        return False
    
    def calculate_similarity_score(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1, where 1 is identical)
        """
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity manually using numpy
        # cosine_similarity = (A · B) / (||A|| * ||B||)
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        # Handle zero vectors to avoid division by zero
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        
        similarity = dot_product / (norm_vec1 * norm_vec2)
        return similarity
    
    def get_similarity_analysis(self, score: float) -> str:
        """
        Convert similarity score to qualitative analysis
        
        Args:
            score: Cosine similarity score
            
        Returns:
            Qualitative analysis string
        """
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
    
    # MODIFIED: Now compares Zoho job against hardcoded jobs
    def compare_zoho_job_to_hardcoded(self, zoho_job_description: str, zoho_job_id: str) -> pd.DataFrame:
        """
        Compare a Zoho job description against hardcoded job descriptions
        
        Args:
            zoho_job_description: Job description fetched from Zoho
            zoho_job_id: The Zoho Job Opening ID
            
        Returns:
            DataFrame with comparison results
        """
        # Get embedding for the Zoho job
        zoho_embedding = self.get_embedding(zoho_job_description)
        if not zoho_embedding:
            return None
        
        # Compare against all hardcoded jobs
        results = []
        for job in self.job_data:
            similarity_score = self.calculate_similarity_score(
                zoho_embedding, 
                job['embedding']
            )
            
            analysis = self.get_similarity_analysis(similarity_score)
            
            results.append({
                'Zoho Job Opening ID': zoho_job_id,
                'Compared To': job['req_id'],
                'Similarity Score': round(similarity_score, 4),
                'Analysis': analysis,
                'Time to Fill': f"{job['time_to_fill']} days"
            })
        
        return pd.DataFrame(results)

# =============================================================================
# STREAMLIT APP - MODIFIED UI
# =============================================================================

def main():
    st.title("🎯 Sourcingology: Job Description Similarity Engine")
    st.subheader("Vector-based job matching with Zoho Recruit integration")
    
    # Initialize the engine
    if 'engine' not in st.session_state:
        st.session_state.engine = JobSimilarityEngine()
    
    engine = st.session_state.engine
    
    # API Key input
    api_key = st.text_input("OpenAI API Key", type="password", 
                           help="Enter your OpenAI API key to enable embeddings")
    
    if api_key:
        openai.api_key = api_key
        
        # NEW: OAuth Helper Section
        st.markdown("---")
        st.subheader("🔐 OAuth Setup Helper")
        
        # Show plan limitations warning
        st.warning("""
        ⚠️ **Zoho Free Tier Limitations:** 
        - API per user count: 0 (may restrict OAuth)
        - Max API Credits/Day: 5,000
        - If you're on the Free tier, OAuth might be limited. Consider upgrading to Standard tier.
        """)
        
        with st.expander("🚀 Generate Refresh Token (Click here if you don't have one)", expanded=False):
            
            # Method 1: Try Self-Client first
            st.markdown("### Method 1: Self-Client (Recommended for Free Tier)")
            st.info("Self-Client authentication sometimes works on free tiers. Try this first!")
            
            self_client_url = f"https://api-console.zoho.com/client/{ZOHO_CLIENT_ID}/refresh"
            
            if st.button("🔗 Try Self-Client Method"):
                st.markdown(f"**[Click here for Self-Client setup →]({self_client_url})**")
                st.info("This will take you directly to generate a refresh token without OAuth flow.")
            
            st.markdown("---")
            
            # Method 2: Full OAuth (if self-client doesn't work)
            st.markdown("### Method 2: Full OAuth Flow")
            st.markdown("""
            **Step 1: Get Authorization Code**
            
            Click the button below to go to Zoho and authorize your app:
            """)
            
            # Use the Streamlit app URL as redirect URI (matches what's registered in Zoho)
            redirect_uri = "https://chatbot-2-nb9zjyd5xozoaclbxrqwh2.streamlit.app"
            
            # Create authorization URL
            auth_url = f"https://accounts.zoho.com/oauth/v2/auth?scope=ZohoRecruit.modules.ALL&client_id={ZOHO_CLIENT_ID}&response_type=code&access_type=offline&redirect_uri={redirect_uri}"
            
            if st.button("🔗 Go to Zoho Authorization (Full OAuth)"):
                st.markdown(f"**[Click here to authorize your app →]({auth_url})**")
                st.info("After authorization, you'll be redirected back to this app. Copy the 'code' parameter from the URL.")
            
            st.markdown("**Step 2: Exchange Code for Refresh Token**")
            
            auth_code = st.text_input(
                "Paste the authorization code here:",
                placeholder="1000.abc123def456...",
                help="Copy the 'code' parameter from the redirect URL"
            )
            
            if st.button("🔄 Get Refresh Token") and auth_code:
                with st.spinner("Exchanging code for refresh token..."):
                    token_url = f"{ZOHO_ACCOUNTS_URL}/oauth/v2/token"
                    
                    payload = {
                        'grant_type': 'authorization_code',
                        'client_id': ZOHO_CLIENT_ID,
                        'client_secret': ZOHO_CLIENT_SECRET,
                        'redirect_uri': redirect_uri,  # Must match the one used in authorization
                        'code': auth_code
                    }
                    
                    try:
                        st.write("🔍 **Debug - Token Exchange Request:**")
                        st.write(f"- Token URL: {token_url}")
                        st.write(f"- Client ID: {ZOHO_CLIENT_ID}")
                        st.write(f"- Redirect URI: {redirect_uri}")
                        st.write(f"- Code: {auth_code[:20]}...")
                        
                        response = requests.post(token_url, data=payload)
                        st.write(f"- Response Status: {response.status_code}")
                        
                        if response.status_code == 200:
                            token_data = response.json()
                            st.write(f"- Response Keys: {list(token_data.keys())}")
                            
                            if 'refresh_token' in token_data:
                                st.success("🎉 Success! Here's your refresh token:")
                                st.code(token_data['refresh_token'], language='text')
                                st.info("**IMPORTANT:** Copy this refresh token and update `ZOHO_REFRESH_TOKEN` in your code!")
                                
                                # Also show access token for immediate testing
                                if 'access_token' in token_data:
                                    st.success("🎯 You also got an access token (valid for 1 hour):")
                                    st.code(token_data['access_token'], language='text')
                            else:
                                st.error(f"❌ No refresh token in response: {token_data}")
                        else:
                            st.error(f"❌ Failed to get tokens (HTTP {response.status_code}):")
                            try:
                                error_data = response.json()
                                st.error(f"Error details: {error_data}")
                                
                                # Check for specific free tier errors
                                if "insufficient" in str(error_data).lower() or "limit" in str(error_data).lower():
                                    st.error("🚨 **This might be a Free Tier limitation!** Consider upgrading to Standard plan.")
                            except:
                                st.error(f"Raw response: {response.text[:500]}...")
                                
                                # Check for HTML error pages (common with free tier limits)
                                if "html" in response.text.lower():
                                    st.error("🚨 **Received HTML error page - likely a Free Tier restriction!**")
                            
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
            st.markdown("---")
            st.markdown("### 💰 Upgrade Recommendation")
            st.info("""
            **If you keep getting errors:**
            1. **Upgrade to Zoho Standard tier** ($12/month) - includes proper API access
            2. **Standard tier gives you:** 100,000 API credits/day + 250 API per user count
            3. **This will likely solve all authentication issues**
            
            [Upgrade your Zoho plan here →](https://www.zoho.com/recruit/pricing.html)
            """)
            
            st.info("💡 **Tip:** Once you get the refresh token, update the `ZOHO_REFRESH_TOKEN` variable at the top of your code, then you can use the main comparison features!")
        
        st.markdown("---")
        st.subheader("📝 Hardcoded Job Descriptions (Job 2 & Job 3)")
        
        # HARDCODED JOB DESCRIPTIONS - Job 2 and Job 3 only
        hardcoded_jobs = {
            "Job 2": {
                "description": """
                Senior Backend Software Engineer - Python Focus
                
                Join our engineering team as a Senior Backend Software Engineer specializing in Python development.
                Looking for someone with 5+ years of backend development experience.
                
                Key Responsibilities:
                - Build and maintain scalable backend applications
                - Develop REST APIs and work with microservices
                - Partner with cross-functional teams including frontend and product
                - Ensure code quality through testing and code reviews
                
                Qualifications:
                - BS in Computer Science or equivalent experience
                - Extensive Python and Django experience
                - Strong database skills with PostgreSQL
                - Cloud platform experience (AWS preferred)
                - Strong analytical and communication abilities
                """,
                "time_to_fill": 76
            },
            
            "Job 3": {
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
        
        # Display hardcoded job descriptions
        for job_name, job_info in hardcoded_jobs.items():
            with st.expander(f"📄 {job_name} - Click to view"):
                st.text_area(f"Job Description ({job_name})", 
                           value=job_info["description"], 
                           height=200, 
                           disabled=True)
                st.write(f"**Time to Fill:** {job_info['time_to_fill']} days")
        
        st.markdown("---")
        
        # NEW: Load hardcoded jobs button (simplified)
        if st.button("🚀 Load Hardcoded Jobs for Comparison"):
            with st.spinner("Generating embeddings for hardcoded jobs..."):
                # Clear existing data
                engine.job_data = []
                
                success_count = 0
                for job_name, job_info in hardcoded_jobs.items():
                    if engine.add_job_description(
                        job_name, 
                        job_info["description"], 
                        job_info["time_to_fill"]
                    ):
                        success_count += 1
                
                if success_count == len(hardcoded_jobs):
                    st.success(f"✅ Successfully generated embeddings for {success_count} hardcoded jobs!")
                else:
                    st.error(f"❌ Only {success_count}/{len(hardcoded_jobs)} embeddings generated successfully")
        
        # NEW: Main comparison section
        if len(engine.job_data) > 0:
            st.markdown("---")
            st.subheader("🔍 Zoho Integration & Comparison")
            
            # Display Zoho configuration
            with st.expander("⚙️ Zoho API Configuration", expanded=True):
                st.write(f"**Target Job Opening ID:** {TARGET_JOB_OPENING_ID}")
                st.write(f"**Zoho Base URL:** {ZOHO_BASE_URL}")
                
                # Credentials validation
                st.subheader("🔐 Credential Check")
                cred_issues = []
                
                if ZOHO_CLIENT_ID == "your_client_id_here":
                    cred_issues.append("❌ Client ID not updated")
                else:
                    st.write(f"✅ Client ID: {ZOHO_CLIENT_ID[:10]}...")
                
                if ZOHO_CLIENT_SECRET == "your_client_secret_here":
                    cred_issues.append("❌ Client Secret not updated")
                else:
                    st.write(f"✅ Client Secret: {ZOHO_CLIENT_SECRET[:10]}...")
                
                if ZOHO_REFRESH_TOKEN == "your_refresh_token_here":
                    cred_issues.append("❌ Refresh Token not updated")
                else:
                    st.write(f"✅ Refresh Token: {ZOHO_REFRESH_TOKEN[:15]}...")
                
                if cred_issues:
                    st.error("**Issues found:**")
                    for issue in cred_issues:
                        st.write(issue)
                    st.warning("⚠️ Please update the credentials at the top of the code!")
                else:
                    st.success("✅ All credentials appear to be configured!")
                
                # Common troubleshooting tips
                st.subheader("🛠️ Troubleshooting Tips")
                st.markdown("""
                **If you're getting a 400 error:**
                1. **Check if your refresh token is still valid** - Refresh tokens can expire
                2. **Verify your Client ID and Client Secret** - Must match your Zoho app exactly
                3. **Ensure your Zoho app has proper permissions** for Recruit API
                4. **Check the data center** - Your token URL might need to be region-specific:
                   - US: `https://accounts.zoho.com/oauth/v2/token`
                   - EU: `https://accounts.zoho.eu/oauth/v2/token` 
                   - IN: `https://accounts.zoho.in/oauth/v2/token`
                   - AU: `https://accounts.zoho.com.au/oauth/v2/token`
                5. **Try generating a new refresh token** if the current one is old
                
                **Common error meanings:**
                - **400 Bad Request**: Usually invalid credentials or malformed request
                - **401 Unauthorized**: Invalid or expired token
                - **403 Forbidden**: Insufficient permissions
                """)
            
            # Test connection button
            if st.button("🧪 Test Zoho Connection", help="Test your Zoho credentials"):
                with st.spinner("Testing Zoho connection..."):
                    access_token = get_zoho_access_token(
                        ZOHO_CLIENT_ID, 
                        ZOHO_CLIENT_SECRET, 
                        ZOHO_REFRESH_TOKEN
                    )
                    
                    if access_token:
                        st.success("🎉 Zoho connection successful!")
                        st.write(f"Access token received: {access_token[:20]}...")
                    else:
                        st.error("❌ Zoho connection failed. Check the debug info above.")
            
            st.markdown("---")
            
            # NEW: Single Compare button (replaces dropdown)
            if st.button("🎯 Compare Job from Zoho", type="primary"):
                with st.spinner("Fetching job from Zoho and calculating similarities..."):
                    
                    # Step 1: Get Zoho access token
                    access_token = get_zoho_access_token(
                        ZOHO_CLIENT_ID, 
                        ZOHO_CLIENT_SECRET, 
                        ZOHO_REFRESH_TOKEN
                    )
                    
                    if not access_token:
                        st.error("❌ Failed to get Zoho access token. Please check your credentials.")
                        return
                    
                    # Step 2: Fetch job description from Zoho
                    zoho_job_description = fetch_job_description_from_zoho(
                        TARGET_JOB_OPENING_ID, 
                        access_token
                    )
                    
                    if not zoho_job_description:
                        st.error(f"❌ Failed to fetch job description for {TARGET_JOB_OPENING_ID}")
                        return
                    
                    # Step 3: Display the fetched job description
                    st.success(f"✅ Successfully fetched job from Zoho!")
                    with st.expander("📄 Fetched Job Description from Zoho", expanded=True):
                        st.text_area(
                            f"Job Description ({TARGET_JOB_OPENING_ID})", 
                            value=zoho_job_description, 
                            height=200, 
                            disabled=True
                        )
                    
                    # Step 4: Run similarity comparison
                    results_df = engine.compare_zoho_job_to_hardcoded(
                        zoho_job_description, 
                        TARGET_JOB_OPENING_ID
                    )
                    
                    if results_df is not None and not results_df.empty:
                        st.subheader("📊 Similarity Results")
                        
                        # Display the results table
                        st.dataframe(
                            results_df, 
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Add insights
                        st.markdown("---")
                        st.subheader("💡 Key Insights")
                        
                        high_matches = results_df[results_df['Similarity Score'] >= 0.85]
                        low_matches = results_df[results_df['Similarity Score'] < 0.30]
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("High Matches (≥0.85)", len(high_matches))
                        
                        with col2:
                            st.metric("Average Similarity", 
                                    f"{results_df['Similarity Score'].mean():.3f}")
                        
                        with col3:
                            st.metric("Low Matches (<0.30)", len(low_matches))
                        
                        # Show most similar job for sourcing insights
                        if not results_df.empty:
                            best_match = results_df.loc[results_df['Similarity Score'].idxmax()]
                            st.info(f"🎯 **Best Match:** {best_match['Compared To']} with {best_match['Similarity Score']:.3f} similarity ({best_match['Analysis']})")
                            st.info(f"📅 **Estimated Time to Fill:** Based on {best_match['Compared To']}, expect approximately {best_match['Time to Fill']}")
                    
        else:
            st.info("👆 Load the hardcoded jobs first to enable comparison")
        
        # Educational section (unchanged)
        if len(engine.job_data) > 0:
            st.markdown("---")
            st.subheader("🧠 How Vector Similarity Works")
            
            st.markdown("""
            **Under the Hood:**
            
            1. **Vectorization**: Each job description is converted into a high-dimensional vector (embedding) using OpenAI's `text-embedding-3-small` model. These vectors capture semantic meaning.
            
            2. **Cosine Similarity**: We calculate the cosine of the angle between two vectors using the formula: `(A·B) / (||A|| × ||B||)`. Values range from -1 to 1, where:
               - **1.0** = Identical content
               - **0.8-0.95** = Very similar content
               - **0.5-0.8** = Moderately similar
               - **< 0.5** = Low similarity
            
            3. **Semantic Understanding**: Unlike keyword matching, embeddings understand context and meaning. "Backend Developer" and "Server-side Engineer" would score highly even without exact word matches.
            
            **Zoho Integration Benefits:**
            - Real-time job description fetching
            - Compare live job postings against historical data
            - Build data-driven sourcing strategies
            
            **Next Steps for Sourcingology:**
            - Scale to 100+ historical job descriptions
            - Use similarity scores to predict time-to-fill
            - Build sourcing strategy recommendations
            """)

if __name__ == "__main__":
    main()
