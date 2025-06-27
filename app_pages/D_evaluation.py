import streamlit as st
import pandas as pd
from src.data_management import download_files_from_gcs


def evaluation_body():
    """
    This function renders the content for the "Evaluation" page.
    It provides an overview of the evaluation process, including metrics,
    results, and insights.
    """
    logo = ("https://res.cloudinary.com/dybts6jei/image/upload/v1750626537/"
            "logo_red_kcrp8n.png")
    st.sidebar.image(logo, width=200)

    Vinyard = ("https://res.cloudinary.com/dybts6jei/image/upload/v1750626558/"
               "dan-meyers-0AgtPoAARtE-unsplash_naudxa.jpg")

    st.image(Vinyard, caption="Photo by Dan Meyers on Unsplash")

    st.title("Evaluation of Model Performance")
    st.write(
        "This section evaluates the performance of the machine learning model "
        "used for personalised wine recommendations."
    )
    st.subheader("White Wine Evaluation")

    st.error(
        "I like white wine, especially Sauvignon Blanc. "
        "I enjoy wines that are fresh and fruity, with a hint of citrus. "
        "I prefer wines that are not too sweet and have a crisp finish."
    )

    st.write("### White Wine Evaluation Results")
    white_cosine = load_cosine_similarity_data()
    cosine_drop_index = white_cosine[["similarity"]].reset_index(drop=True)
    cosine_drop_index.index = cosine_drop_index.index + 1
    st.dataframe(cosine_drop_index)
    # with analysis:
    #     st.dataframe(cosine_drop_index.describe())

    st.write("### Analysis of Results")
    st.write("""
* The mean similarity score for the white wine recommendations is **0.76**.
* This indicates the semantic meaning of the wine descriptions is well-aligned
with user preferences.
* With a **0.08** variance, the scores are relatively consistent.
* The model's ability to recommend wines that match user preferences is
                 evident, with a high average similarity score.
""")

    st.write("### Top White Wines")
    white_top_wines = load_dataframe()
    st.dataframe(white_top_wines)
    st.markdown("""
The algorithm performed well in identifying subtle nuances, particularly
in detecting a **touch of sweetness on the finish.** This demonstrates its
ability to comprehend the semantic meaning of **not too sweet.**

It frequently matched descriptions based on specific words, such as **citrus.**
However, it also successfully identified De Loach using related terms like
**lime,** **lemon,** and **pear.** Furthermore, it presented a variety,
including **Sémillon, aged in neutral French oak,** which aligns with nuances
associated with semi-dry wines.

Every wine selected by the algorithm had ratings of either
**very good quality** or **excellent.** This suggests that
consumers are likely to enjoy the recommended options.

The Ca' Momi, though a sparkling blend rather than a white wine, matched the
description well. It was neither overly fruity nor excessively sweet, aligning
closely with the intended profile, even if not the exact category expected.
This offers the end-user to try something new based on their taste profile.
                """)
    st.error("The jupyter notebook and README.md files contain more details on"
             "the evaluation process.")
    st.subheader("Conclusion")
    st.markdown("""
The analysis of the content-based wine recommendation system, using
**SentenceTransformer** for description similarity across red, white, and
sparkling wine categories, reveals significant strengths in semantic
understanding of both user and professional descriptions of wine.

### Key Strengths Observed:

* Across all wine types, the algorithm demonstrated a strong ability to
  comprehend and match the positive semantic meaning of user input. It
  successfully identified wines with desired traits like **dark fruits** in
  the **luxurious black cherry and dark chocolate flavors**, **crisp fruit
  flavors,** **tangy**, **citrus finish**, and **light and refreshing.**

* The system had a nuanced approach to selecting wine, being able to
  understand subtle user preferences, such as the white wine user's
  **not too sweet** and being able to effectively match with wines having a
  **hint of sweetness** or being **moderately sweet.** This showcased its
  capacity to go beyond exact keyword matching.

* Although the dataset was already skewed towards good quality wine, it was
  still able to only select high ratings which is a valuable outcome for user
  satisfaction.

* Showed promise in recommending wines that semantically align with a user's
  taste profile, potentially leading to discoveries from regions or varieties
  not explicitly mentioned by the user, thereby expanding their exploration.

### Identified Challenges and Areas for Refinement:

* The model struggled to select red wines, especially when handling negative
  constraints and co-occurrence bias, particularly when trying to find
  "not too tannic" red wines. Due to the natural co-occurrence of terms like
  **bold,** **full-bodied,** **age-worthy** often appearing with **tannic**,
  the algorithm tended to prioritise the abundance of positive semantic
  matches, sometimes overlooking the singular negative preference. This bias
  in the current semantic similarity model needs to be addressed.

### Recommendations for Improvement:

A hybrid recommendation logic should be developed that integrates the
categorical features alongside the description embeddings. With pre-filtering
features that would constrain its outputs to **sparkling wine** this would
improve its reliability. Nevertheless, in its current form, customers can
venture out of their comfort zone and find wines that are similar to their
tastes, which was one of the business requirements.

The model has only been evaluated using pure analysis of its outputs and
cosine similarity. For a more robust evaluation, ground truth from real users
should be used to improve its ability to predict what wines people would
enjoy.

As the project aimed to not waste time drinking bad wine, the system has
proved that it can recommend wines that are of a good quality, will help the
user expand their repertoire, and at the same time provide predictability of
its ability to match the user's tastes.
""")


def load_cosine_similarity_data():

    try:
        download_files_from_gcs(
            bucket_name="vinefind",
            source_blob_name="datasets/evaluation/white_cosine.pkl",
            destination_file_name=(
                "VineFind_v2/outputs/datasets/evaluation/white_cosine.pkl"
            )
        )
        white_cosine = pd.read_pickle(
            "VineFind_v2/outputs/datasets/evaluation/white_cosine.pkl"
        )
    except Exception as e:
        st.error(
            f"🍷 Oops! We couldn't fetch the wine dataset needed to "
            f"explore the data. Please try again later. Error: {e}"
        )
        return None

    return white_cosine


def load_dataframe():

    try:
        download_files_from_gcs(
            bucket_name="vinefind",
            source_blob_name="datasets/evaluation/white_top_wines.pkl",
            destination_file_name=(
                "VineFind_v2/outputs/datasets/evaluation/white_top_wines.pkl"
            )
        )
        white_top_wines = pd.read_pickle(
            "VineFind_v2/outputs/datasets/evaluation/white_top_wines.pkl"
        )
    except Exception as e:
        st.error(
            f"🍷 Oops! We couldn't fetch the wine dataset needed to "
            f"explore the data. Please try again later. Error: {e}"
        )
        return None

    return white_top_wines


def sidebar_body():
    """
    This function renders the sidebar content for the "Project Summary" page.
    It provides a brief overview of the project and its objectives.
    """
    logo = ("https://res.cloudinary.com/dybts6jei/image/upload/v1750626537/"
            "logo_red_kcrp8n.png")
    st.sidebar.image(logo, width=200)
    st.write("VineFind")
