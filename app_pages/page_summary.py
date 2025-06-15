import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.data_management import load_pkl_file, load_assets
import time


def page_summary_body():
    """
    This function renders the content for the "Project Summary" page.
    It provides an overview of the project, including objectives, data sources,
    and key findings.
    """

    st.image(
        "VineFind/assets/d-a-v-i-d-s-o-n-l-u-n-a-hupBI0Doj9o-unsplash.jpg",
        caption="Photo taken by David Luna on Unsplash"
    )

    st.title("Project Summary")

    st.write(
        "Life’s too short for a bad bottle of wine — and too short to keep "
        "grabbing the same one just because it’s familiar. "
    )
    st.write(
        "This tool helps you discover new wines based on what you like. "
        "Describe your favorite wine, think"
    )

    st.image("VineFind/assets/fruit-flavors-red-white-wine-folly-infographic.jpg", caption="Created by Folly Wine")

    st.subheader("How to Use This Tool")

    st.markdown(
        "1. **Input**: You describe your favorite wine.\n"
        "2. **Embedding**: The input is converted into a numerical "
        "embedding using a pre-trained model.\n"
        "3. **Similarity Calculation**: The embedding is compared "
        "against a dataset of wine reviews to find the most similar wines.\n"
        "4. **Recommendations**: The top 10 similar wines are displayed."
    )

    st.write(
        "Write a few sentences about your favorite wine, "
        "and the tool will suggest similar wines based on your description. "
        "The more detailed your description, the better the recommendations. "
    )

    st.markdown(
        "*e.g. I like white wine, especially Sauvignon Blanc* "
        "*from New Zealand. I enjoy wines that are fresh and fruity,* "
        "*with a hint of citrus. I prefer wines that are not too sweet* "
        "*and have a crisp finish.* "
    )

    with st.form(
        key="user_input", clear_on_submit=True, enter_to_submit=False
    ):

        user_input = st.text_area(
            "Describe your favorite wine",
            height=100
        )
        submit = st.form_submit_button("Get Recommendations")
        if submit and user_input:
            with st.spinner(
                "Finding your perfect wine..., please wait..."
            ):
                time.sleep(2)
                st.success("Your recommendations are on their way!")
            try:
                similarities_df = user_embeddings(user_input)
                top_wines = compute(similarities_df)
                display_recommendations(top_wines)
            except Exception as e:
                st.error(
                    f"An error occurred while processing your request: {e}"
                )

    st.subheader("💬 I'd love your feedback!")
    st.write(
        "📝 This is still a work in progress, and your feedback is "
        "valuable. If something felt off or the recommendations were "
        "spot-on, please let me know! Your input will help fine-tune "
        "the model and make it better for everyone."
    )
    st.markdown("Please [click here to provide feedback]"
                "(https://forms.gle/2MymytdLu3E3bPKC9) ")

    return user_input


def user_embeddings(user_input):
    """
    This function generates embeddings for the user input.
    In a real application, this would call an embedding model.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    x_train = load_pkl_file(
        'VineFind/VineFind_v1/outputs/datasets/encoded/train/model_b/'
        'embeddings_train_model_b.pkl'
    )
    user_input_embedding = model.encode([user_input])
    x_train_embed_col = [
        em for em in x_train.columns if em.startswith('embedding')
    ]
    x_train_embed = x_train[x_train_embed_col].values.tolist()
    similarities = cosine_similarity(user_input_embedding, x_train_embed)
    similarities = similarities.flatten()
    similarities_df = pd.DataFrame({
        'similarity': similarities,
        'index': x_train.index
    })
    return similarities_df.sort_values(by='similarity',
                                       ascending=False)


def compute(similarities_df):
    """
    This function displays the top 10 recommendations based on the user's
    input.
    """
    x_train_original = load_pkl_file(
        'VineFind/VineFind_v1/outputs/datasets/cleaned/display_dataframe.pkl'
    )
    st.subheader("Top 10 Recommendations")
    st.write("Double click on a row to see the full description.")
    top_10 = 10
    top_similarities = similarities_df.sort_values(by='similarity',
                                                   ascending=False).head(
                                                       top_10)
    top_similarities = top_similarities['index'].values
    top_wines = x_train_original.loc[top_similarities]
    return top_wines


def clean_column_names(top_wines):
    """
    This function cleans the column names of a DataFrame by removing
    leading and trailing whitespace.
    """
    top_wines.columns = [
        (col[0] if isinstance(col, tuple) else col).title()
        for col in top_wines.columns
    ]
    return top_wines


def display_recommendations(top_wines):
    """
    This function displays the top 10 wine recommendations in a Streamlit
    DataFrame.
    """
    top_wines = clean_column_names(top_wines)
    top_wines = top_wines[['Winery', 'Description', 'Variety', 'Country',
                           'Province']].head(10)
    #'Price' to be added later
    recommendations = st.dataframe(top_wines, height=500, hide_index=True,
                                   )
    return recommendations
