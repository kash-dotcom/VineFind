import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


def load_pkl_file(file_path):
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


def load_assets(*path_parts):
    return os.path.join(*path_parts)
