import openai
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import os

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
            time_to_fill: Time to fill in days (placeholder for now)
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
        # Reshape for sklearn's cosine_similarity function
        emb1 = np.array(embedding1).reshape(1, -1)
        emb2 = np.array(embedding2).reshape(1, -1)
        
        similarity = cosine_similarity(emb1, emb2)[0][0]
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
    
    def compare_job_descriptions(self, target_req_id: str) -> pd.DataFrame:
        """
        Compare a target job description against all others in the dataset
        
        Args:
            target_req_id: The requisition ID to compare against others
            
        Returns:
            DataFrame with comparison results
        """
        # Find the target job
        target_job = None
        for job in self.job_data:
            if job['req_id'] == target_req_id:
                target_job = job
                break
        
        if not target_job:
            st.error(f"Job {target_req_id} not found in dataset")
            return None
        
        # Compare against all other jobs
        results = []
        for job in self.job_data:
            if job['req_id'] != target_req_id:  # Don't compare to itself
                similarity_score = self.calculate_similarity_score(
                    target_job['embedding'], 
                    job['embedding']
                )
                
                analysis = self.get_similarity_analysis(similarity_score)
                
                results.append({
                    'Requisition ID': target_job['req_id'],
                    'Compared To': job['req_id'],
                    'Similarity Score': round(similarity_score, 4),
                    'Analysis': analysis,
                    'Time to Fill': f"{job['time_to_fill']} days"
                })
        
        return pd.DataFrame(results)

# Streamlit App Implementation
def main():
    st.title("🎯 Sourcingology: Job Description Similarity Engine")
    st.subheader("Vector-based job matching prototype")
    
    # Initialize the engine
    if 'engine' not in st.session_state:
        st.session_state.engine = JobSimilarityEngine()
    
    engine = st.session_state.engine
    
    # API Key input
    api_key = st.text_input("OpenAI API Key", type="password", 
                           help="Enter your OpenAI API key to enable embeddings")
    
    if api_key:
        openai.api_key = api_key
        
        st.markdown("---")
        st.subheader("📝 Sample Job Descriptions for Testing")
        
        # Sample job descriptions for testing
        job_descriptions = {
            "REQ001": """
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
            
            "REQ002": """
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
            
            "REQ003": """
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
            """
        }
        
        # Display job descriptions
        for req_id, description in job_descriptions.items():
            with st.expander(f"📄 {req_id} - Click to view"):
                st.text_area(f"Job Description ({req_id})", 
                           value=description, 
                           height=200, 
                           disabled=True)
        
        st.markdown("---")
        
        # Load sample data button
        if st.button("🚀 Load Sample Data & Generate Embeddings"):
            with st.spinner("Generating embeddings... This may take a moment."):
                # Clear existing data
                engine.job_data = []
                
                # Add job descriptions with different time-to-fill values for variety
                time_to_fill_data = {"REQ001": 89, "REQ002": 76, "REQ003": 45}
                
                success_count = 0
                for req_id, description in job_descriptions.items():
                    if engine.add_job_description(req_id, description, time_to_fill_data[req_id]):
                        success_count += 1
                
                if success_count == len(job_descriptions):
                    st.success(f"✅ Successfully generated embeddings for {success_count} job descriptions!")
                else:
                    st.error(f"❌ Only {success_count}/{len(job_descriptions)} embeddings generated successfully")
        
        # Comparison section
        if len(engine.job_data) > 0:
            st.markdown("---")
            st.subheader("🔍 Similarity Analysis")
            
            # Select target job for comparison
            target_job = st.selectbox(
                "Select the job description to compare against others:",
                options=[job['req_id'] for job in engine.job_data],
                index=0
            )
            
            if st.button("🎯 Run Similarity Analysis"):
                with st.spinner("Calculating similarities..."):
                    results_df = engine.compare_job_descriptions(target_job)
                    
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
                        
                        # Explain the concept
                        st.markdown("---")
                        st.subheader("🧠 How Vector Similarity Works")
                        
                        st.markdown("""
                        **Under the Hood:**
                        
                        1. **Vectorization**: Each job description is converted into a high-dimensional vector (embedding) using OpenAI's `text-embedding-3-small` model. These vectors capture semantic meaning.
                        
                        2. **Cosine Similarity**: We calculate the cosine of the angle between two vectors. Values range from -1 to 1, where:
                           - **1.0** = Identical content
                           - **0.8-0.95** = Very similar content
                           - **0.5-0.8** = Moderately similar
                           - **< 0.5** = Low similarity
                        
                        3. **Semantic Understanding**: Unlike keyword matching, embeddings understand context and meaning. "Backend Developer" and "Server-side Engineer" would score highly even without exact word matches.
                        
                        **Next Steps for Sourcingology:**
                        - Scale to 100+ historical job descriptions
                        - Use similarity scores to predict time-to-fill
                        - Build sourcing strategy recommendations
                        """)
                    
        else:
            st.info("👆 Load the sample data first to see the similarity analysis")

if __name__ == "__main__":
    main()
