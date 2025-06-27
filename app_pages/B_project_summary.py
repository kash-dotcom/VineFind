import streamlit as st
# from src.data_management import load_file


def project_summary_body():
    """
    This function renders the content for the "Project Summary" page.
    It provides an overview of the project, including objectives, data sources,
    and key findings.
    """
    logo = ("https://res.cloudinary.com/dybts6jei/image/upload/v1750626537/"
            "logo_red_kcrp8n.png")
    st.sidebar.image(logo, width=200)

    cheese_and_wine = (
        "https://res.cloudinary.com/dybts6jei/image/upload/"
        "v1750626552/camille-brodard-f6Wpz1QPFZI-unsplash_ybqbgl.jpg"
    )
    st.image(cheese_and_wine)
    st.caption("Photo by Camille Brodard on Unsplash")

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


def sidebar_body():
    """
    This function renders the sidebar content for the "Project Summary" page.
    It provides a brief overview of the project and its objectives.
    """
    logo = ("https://res.cloudinary.com/dybts6jei/image/upload/v1750626537/"
            "logo_red_kcrp8n.png")
    st.sidebar.image(logo, width=200)
    st.write("VineFind")
