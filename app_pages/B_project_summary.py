import streamlit as st
import pandas as pd
from src.data_management import load_file


def project_summary_body():
    """
    This function renders the content for the "Project Summary" page.
    It provides an overview of the project, including objectives, data sources,
    and key findings.
    """


with st.expander("Business Case"):
    st.title("Business Case: Personalised Product Recommendations")
    st.markdown(
        "This project aims to enhance customer experience by providing "
        "personalized product recommendations using predictive analytics."
    )

with st.expander("Learning Methods Overview"):
    st.title('Machine Learning Methods')
    st.write("**SentenceTransformer**: Processes and understands "
             "textual data based on user input")

    # st.write("**Nearest Neighbor**: Classification/regression algorithm",
    #     "PCA (Principal Component Analysis): Dimensionality reduction",
    #     "KMeans:" "Clustering algorithm",
    #     "Train-test split: Evaluates model performance",
    #     "Scatter plots:" "Data visualization",
    #     "Classification report" "Assesses model performance metrics "
    #     " (precision, recall, f1-score)"
    # )
    st.write()

st.write("### Key Aspects of the Business Case")
business_case = {
    "Aim": (
        "Enhance customer experience by providing personalized "
        "product recommendations"
    ),
    "Method": "Predictive analytics using machine learning algorithms",
    "Outcome": "Accurate and relevant product suggestions",
    "Metrics": (
        "Success measured by focus group feedback and Nearest Neighbor "
        "score (target >90%, current 70% accuracy)"
    ),
    "Relevance": "Helps customers find products matching their"
                    "preferences",
    "Heuristics & Data": (
        "Utilizes customer input and historical data for training"
    )
}
