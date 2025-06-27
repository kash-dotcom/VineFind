import streamlit as st
import pandas as pd
from src.data_management import download_files_from_gcs


def evaluation_body():
    """
    This function renders the content for the "Evaluation" page.
    It provides an overview of the evaluation process, including metrics,
    results, and insights.
    """

    st.title("Evaluation of Model Performance")
    st.write(
        "This section evaluates the performance of the machine learning model "
        "used for personalised wine recommendations."
    )
    st.subheader("White Wine Evaluation")

    st.warning(
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

    # # Load evaluation data
    # evaluation_data = download_files_from_gcs(
    #     bucket_name="vinefind-bucket",
    #     source_blob_name="evaluation_data.csv",
    #     destination_file_name="evaluation_data.csv"
    # )

    # if evaluation_data is not None:
    #     df = pd.read_csv("evaluation_data.csv")
    #     st.dataframe(df)

    #     st.subheader("Evaluation Metrics")
    #     st.write("The model's performance is evaluated using various metrics
    # such as accuracy, precision, recall, and F1-score.")

    #     # Display metrics
    #     metrics = {
    #         "Accuracy": df["accuracy"].mean(),
    #         "Precision": df["precision"].mean(),
    #         "Recall": df["recall"].mean(),
    #         "F1 Score": df["f1_score"].mean()
    #     }

    #     st.json(metrics)

    #     st.subheader("Insights")
    #     st.write(
    #         "The model shows promising results with an overall accuracy of "
    #         f"{metrics['Accuracy']:.2f}. Further
    # improvements can be made by "
    #         "tuning hyperparameters and incorporating more
    #  diverse training data."
    #     )
    # else:
    #     st.error("Failed to load evaluation data.")
