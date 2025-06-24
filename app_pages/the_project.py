import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.data_management import load_pkl_file
import os
# import io
import base64
import time
from google.cloud import storage
import json


# code sourced from Gemini
def download_files_from_gcs(bucket_name, source_blob_name,
                            destination_file_name):
    try:
        storage_client = None  # Initialise to None

        # 1. Try to get credentials from Streamlit secrets (for local
        # development/Streamlit Cloud)
        # Check if running on Heroku based on DYNO environment variable
        is_heroku = "DYNO" in os.environ

        if is_heroku:
            st.info("Running on Heroku. Using credentials from GCP_SERVICE_ACCOUNT_KEY environment variable.")
            if "GCP_SERVICE_ACCOUNT_KEY" in os.environ:
                encoded_key = os.environ["GCP_SERVICE_ACCOUNT_KEY"]
                try:
                    decoded_key_json = base64.b64decode(encoded_key).decode('utf-8')
                    credentials_info = json.loads(decoded_key_json)
                    storage_client = storage.Client.from_service_account_info(credentials_info)
                except (base64.binascii.Error, json.JSONDecodeError) as decode_error:
                    st.error(f"Error decoding or parsing GCP_SERVICE_ACCOUNT_KEY: {decode_error}")
                    return None
            else:
                st.error("GCP_SERVICE_ACCOUNT_KEY environment variable not found on Heroku.")
                st.stop()
                return None
        else: # Not on Heroku (local or Streamlit Cloud)
            st.info("Not on Heroku. Using credentials from Streamlit secrets (local/Streamlit Cloud).")
            if "connections" in st.secrets and "gcs" in st.secrets["connections"]:
                credentials_info = st.secrets["connections"]["gcs"]
                storage_client = storage.Client.from_service_account_info(credentials_info)
            else:
                st.error("GCS credentials not found in Streamlit secrets for local/Streamlit Cloud.")
                st.stop()
                return None

        if storage_client is None:
            st.error("Failed to initialize Google Cloud Storage client. No valid credentials found.")
            return None

        # --- Remainder of your function (no changes needed here) ---
        destination_dir = os.path.dirname(destination_file_name)
        if destination_dir and not os.path.exists(destination_dir):
            st.info(f"Creating local directory on dyno: {destination_dir}")
            os.makedirs(destination_dir, exist_ok=True)

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)

        if not blob.exists():
            st.error(f"Error: Blob '{source_blob_name}' does not exist in bucket '{bucket_name}'. Please check the file path in GCS.")
            return None

        st.info(f"Attempting to download '{source_blob_name}' to '{destination_file_name}' on dyno...")
        blob.download_to_filename(destination_file_name)
        st.success(f"File '{source_blob_name}' downloaded to '{destination_file_name}' successfully on dyno!")
        return destination_file_name
    except Exception as e:
        st.error(f"An unexpected error occurred during GCS download: {e}")
        st.exception(e)
        return None


def the_project_body():
    """
    This function renders the content for the "Project Summary" page.
    It provides an overview of the project, including objectives, data sources,
    and key findings.
    """

    img_life_too_short = (
        "https://res.cloudinary.com/dybts6jei/image/upload/"
        "v1750626554/d-a-v-i-d-s-o-n-l-u-n-a-hupBI0Doj9o-unsplash_qdjyb4.jpg"
        )
    st.image(img_life_too_short)
    st.caption("Photo taken by David Luna on Unsplash")

    st.title("Uncork the Unexpected: Discover Your Next Favorite Wine")
    sidebar_body()
    st.write(
        "Life’s too short for a bad bottle of wine — and too short to keep "
        "grabbing the same one just because it’s familiar. "
    )
    st.write(
        "This tool helps you discover new wines based on what you like. "
        "Describe your favorite wine, and it will suggest similar wines "
    )

    st.markdown("Right now, you can’t pick a price range for your wine "
                "recommendations. We’re working on it, so stay tuned "
                "for future updates!"
                )
    img_flavor_glass = (
        "https://res.cloudinary.com/dybts6jei/image/"
        "upload/v1750626536/fruit-flavors-red-white-wine-folly-infographic"
        "_ufuzco.jpg"
        )
    st.image(img_flavor_glass)
    st.caption("A red and wine flavor chart created by Folly Wine")

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
    st.write("For example...")
    img_wine_example = (
        "https://res.cloudinary.com/dybts6jei/image/upload/"
        "v1750626539/wine_quote1_ov6esv.png")
    st.image(img_wine_example,
             caption="An example wine description from a wine lover",
             )

    with st.form(
        key="user_input", clear_on_submit=False, enter_to_submit=True
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
                top_wines = compute(similarities_df, user_input)
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


def sidebar_body():
    """
    This function renders the sidebar content for the "Project Summary" page.
    It provides a brief overview of the project and its objectives.
    """
    st.sidebar.title("Rating & Price Guide")

    df_quality = pd.DataFrame({
        'Key': [
            '🍷',
            '🍷🍷',
            '🍷🍷🍷',
            '🍷🍷🍷🍷',
            '🍷🍷🍷🍷🍷'
        ],
        'Quality': [
            'Fair to poor quality, not recommended',
            'Good quality, drinkable',
            'Very good quality, worth trying',
            'Excellent quality, highly recommended',
            'Exceptional quality, a must-try'
        ],
        'Wine Ranking': [
            'Below 80',
            '80 - 85',
            '85 - 90',
            '90 - 95',
            'Over 95'
        ],

    })
    st.sidebar.dataframe(df_quality)

    price_buyer_df = pd.DataFrame({
        'Key': [
            '💲',
            '💲💲',
            '💲💲',
            '💲💲',
            '💲💲💲',
            '💲💲💲',
            '💲💲💲💲',
        ],
        'Buyers': [
            'Casual drinkers, everyday wines',
            'Budget-conscious enthusiasts',
            'Dinner party hosts, wine-lovers',
            'Wine enthusiasts seeking quality',
            'Gifts, special occasions, fine dining',
            'Serious collectors, high-end gifts',
            'Investors, connoisseurs, and collectors',
        ],

        'Price Range': [
            "Below $9",
            "$10 - $19",
            "$20 - $49",
            "$50 - $99",
            "$100 - $499",
            "$500 - $999",
            "$,1000 - $2999"
        ]

    })
    st.sidebar.dataframe(price_buyer_df)


def user_embeddings(user_input):
    """
    This function generates embeddings for the user input.
    In a real application, this would call an embedding model.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')

    try:
        download_files_from_gcs(
            bucket_name="vinefind",
            source_blob_name="datasets/encoded/description.pkl",
            destination_file_name=(
                "VineFind_v2/outputs/datasets/encoded/description.pkl"
            )
        )
        df = load_pkl_file(
            "VineFind_v2/outputs/datasets/encoded/description.pkl"
        )
    except Exception as e:
        st.error(
            f"🍷 Oops! We couldn't fetch the wine dataset needed to "
            f"recommend your perfect bottle. "
            f"Please try again later"
            f"Error: {e}"
        )
        return None

    user_input_embedding = model.encode([user_input])

    x_embed_col = [col for col in df.columns if col.startswith('embedding')]
    x_embed = df[x_embed_col].values

    similarities = cosine_similarity(user_input_embedding, x_embed)

    similarities = similarities.flatten()
    similarities_df = pd.DataFrame({
        'similarity': similarities,
        'index': df.index
    })

    return similarities_df


def compute(similarities_df, user_input):
    """
    This function displays the top 10 recommendations based on the user's
    input.
    """
    local_path = (
        "VineFind_v2/outputs/datasets/cleaned/"
        "display_dataframe.pkl"
    )

    try:
        df_original = download_files_from_gcs(
            bucket_name="vinefind",
            source_blob_name=(
                "datasets/cleaned/display_dataframe.pkl"
            ),
            destination_file_name=local_path
        )

        df_original = load_pkl_file(local_path)

    except Exception:
        st.error(
            "🍷 Oops! We couldn't fetch the wine dataset needed to "
            "recommend your perfect bottle. "
            "Please try again later"
        )
        st.write("Error: Unable to load the wine dataset.")
        return None

    st.subheader("Top 10 Recommendations")
    st.write(user_input)
    top_10 = 10
    top_similarities = (
        similarities_df.sort_values(by='similarity', ascending=False)
        .head(top_10)
    )

    top_similarities = top_similarities['index'].values

    top_wines = df_original.loc[top_similarities]

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
    top_wines = top_wines.reset_index()
    if 'id' in top_wines.columns:
        top_wines = top_wines.drop(columns=['id'])
    top_wines = top_wines[['Title', 'Description', 'Variety',
                          'Province', 'Buyer Price', 'Rating']].head(10)
    # 'Price' to be added later
    top_wines.index = top_wines.index + 1
    recommendations = st.table(top_wines)

    print(f"Recommendations: {recommendations}")
    return recommendations
