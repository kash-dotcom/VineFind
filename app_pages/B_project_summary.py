import streamlit as st
# from src.data_management import load_file


def project_summary_body():
    """
    This function renders the content for the "Project Summary" page.
    It provides an overview of the project, including objectives, data sources,
    and key findings.
    """

    st.title(
        "Project Summary: VineFind - A content-based wine "
        "recommendation system"
    )
    st.write(
        "To help users discover wines that match their taste preferences "
        "by analysing both user input and expert wine descriptions. "
        "The goal is to reduce the time and risk associated with "
        "selecting new wines, ensuring users are more likely to "
        "enjoy their purchases."
    )

    st.subheader("Learning Methods")
    st.write(
        "The system uses semantic similarity using a pre-trained "
        "Sentence Transformers fpr deep learning for natuarl language"
        "processing to encode both the user descriptions of the wines "
        "they enjoy and professional wine descriptions. "
        "Cosine similarity is then used to match user input to the "
        "most relevant wines in the dataset."
    )
    st.write(
        "Exploratory Data Analysis (EDA) using profiling tools, "
        "visualisation (boxplots, word clouds), and statistical "
        "tests (e.g., chi-squared for price/points association)."
    )
    st.subheader("Ideal Outcome")
    st.write(
        "The ideal outcome is to provide users with a list of wines "
        "that closely match their preferences, enhancing their "
        "wine selection experience and increasing the likelihood "
        "of satisfaction with their purchases. With a nuanced "
        "understanding of user preferences, the system aims to "
        "closely match wines to user tastes, even when "
        "they are not explicitly stated. "
    )
    st.subheader("Success and Failure Metrics")

    st.write(
        "Success: High average cosine similarity (70%) between "
        "user input and recommended wine descriptions; "
        "positive user feedback; high ratings for recommended "
        "wines; diversity in recommendations that still match "
        "user intent."
    )

    st.write(
        "Failure: Low cosine similarity (below 50%); "
        "negative user feedback; recommendations that do not "
        "align with user preferences; lack of diversity in "
        "recommendations leading to repetitive suggestions."
    )

    st.subheader("Model Output & User Relevance")
    st.write(
        "The model outputs a list of top wine "
        "recommendations, each with a similarity"
        "score, description, and key attributes "
        "(winery, variety, price range, rating). "
        "This output is directly relevant to users, "
        "enabling them to make informed, personalised "
        "wine choices and encouraging exploration beyond "
        "their usual selections."
    )

    st.subheader("Heuristics and Data")
    st.markdown("""

* Over 160,000 expert wine reviews, cleaned and pre-processed. That was
                reduced to 14,000.
* The description feature was embedded using Sentence Transformers.
* Removal of duplicates based on wine descriptions.
* Imputation strategies and disclaimers for missing price data.

                """)

# with st.expander("Business Case"):
#     st.title("Business Case: Personalised Product Recommendations")
#     st.markdown(
#         "This project aims to enhance customer experience by providing "
#         "personalized product recommendations using predictive analytics."
#     )

# with st.expander("Learning Methods Overview"):
#     st.title('Machine Learning Methods')
#     st.write("**SentenceTransformer**: Processes and understands "
#              "textual data based on user input")

#     # st.write("**Nearest Neighbor**: Classification/regression algorithm",
#     #     "PCA (Principal Component Analysis): Dimensionality reduction",
#     #     "KMeans:" "Clustering algorithm",
#     #     "Train-test split: Evaluates model performance",
#     #     "Scatter plots:" "Data visualization",
#     #     "Classification report" "Assesses model performance metrics "
#     #     " (precision, recall, f1-score)"
#     # )
#     st.write()

# st.write("### Key Aspects of the Business Case")
# business_case = {
#     "Aim": (
#         "Enhance customer experience by providing personalized "
#         "product recommendations"
#     ),
#     "Method": "Predictive analytics using machine learning algorithms",
#     "Outcome": "Accurate and relevant product suggestions",
#     "Metrics": (
#         "Success measured by focus group feedback and Nearest Neighbor "
#         "score (target >90%, current 70% accuracy)"
#     ),
#     "Relevance": "Helps customers find products matching their"
#                     "preferences",
#     "Heuristics & Data": (
#         "Utilizes customer input and historical data for training"
#     )
# }
