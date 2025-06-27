import streamlit as st

from app_pages.multipage import MultiPage
from dotenv import load_dotenv
import cloudinary
import os

# load pages scripts
from app_pages.A_the_pineline import the_project_body
from app_pages.B_project_summary import project_summary_body
from app_pages.C_data_exploration import data_exploration_page
from app_pages.D_evaluation import evaluation_body

st.set_page_config(
            page_title="VineFind",
            page_icon="🍷")

app = MultiPage(app_name="VineFind ")


# Create an instance of the app

# Add your app pages here using .add_page()
app.add_page("The Pipeline", the_project_body)
app.add_page("Project Summary", project_summary_body)
app.add_page("Data Exploration", data_exploration_page)
app.add_page("Evaluation", evaluation_body)


load_dotenv()

cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
api_key = os.getenv("CLOUDINARY_API_KEY")
api_secret = os.getenv("CLOUDINARY_API_SECRET")

cloudinary.config(
    cloud_name=cloud_name,
    api_key=api_key,
    api_secret=api_secret
)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    r"C:/Users/purpk/OneDrive/Documents/Coding/VineFind/VineFind/"
    r"gcs_key_vinefind.json"
)

app.run()  # Run the  app
