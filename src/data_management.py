import streamlit as st
import joblib
import os
import base64
from google.cloud import storage
import json


def load_file(file_path):
    """
    Load a pickle file from the specified path.

    Parameters:
    file_path (str): The path to the pickle file.

    Returns:
    object: The object loaded from the pickle file.
    """
    try:
        return joblib.load(file_path)
    except Exception as e:
        st.error(f"Error loading file {file_path}: {e}")
        return None


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
            st.info(
                "Running on Heroku. Using credentials from "
                "GCP_SERVICE_ACCOUNT_KEY environment variable."
            )
            if "GCP_SERVICE_ACCOUNT_KEY" in os.environ:
                encoded_key = os.environ["GCP_SERVICE_ACCOUNT_KEY"]
                try:
                    decoded_bytes = base64.b64decode(encoded_key)
                    decoded_key_json = decoded_bytes.decode('utf-8')
                    credentials_info = json.loads(decoded_key_json)
                    storage_client = storage.Client.from_service_account_info(
                        credentials_info
                    )
                except (
                    base64.binascii.Error, json.JSONDecodeError
                ):
                    return None
            else:
                st.error(
                    "GCP_SERVICE_ACCOUNT_KEY environment variable not "
                    "found on Heroku."
                )
                st.stop()
                return None
        else:  # Not on Heroku (local or Streamlit Cloud)
            if (
                "connections" in st.secrets
                and "gcs" in st.secrets["connections"]
            ):
                credentials_info = st.secrets["connections"]["gcs"]
                storage_client = storage.Client.from_service_account_info(
                    credentials_info
                )
            else:
                st.error(
                    "GCS credentials not found in Streamlit secrets for "
                    "local/Streamlit Cloud."
                )
                st.stop()
                return None

        if storage_client is None:
            st.error(
                "Failed to initialize Google Cloud Storage client. "
                "No valid credentials found."
            )
            return None

        # --- Remainder of your function (no changes needed here) ---
        destination_dir = os.path.dirname(destination_file_name)
        if destination_dir and not os.path.exists(destination_dir):
            st.info(f"Creating local directory on dyno: {destination_dir}")
            os.makedirs(destination_dir, exist_ok=True)

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)

        if not blob.exists():
            st.error(
                f"Error: Blob '{source_blob_name}' does not exist in bucket "
                f"'{bucket_name}'. Please check the file path in GCS."
            )
            return None

        blob.download_to_filename(destination_file_name)
        return destination_file_name
    except Exception as e:
        st.error(f"An unexpected error occurred during GCS download: {e}")
        st.exception(e)
        return None
