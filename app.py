import streamlit as st

st.set_page_config(
            page_title="VineFind",
            page_icon="🍷")

from app_pages.multipage import MultiPage
from dotenv import load_dotenv
import cloudinary
import os
# import json
from app_pages.A_the_pineline import the_project_body
from app_pages.C_data_exploration import data_exploration_page
from app_pages.D_evaluation import evaluation_body
# from app_pages.B_project_summary import project_summary_body


app = MultiPage(app_name="VineFind ")

# load pages scripts

# from app_pages.page_churned_customer_study
# import page_churned_customer_study_body
# from app_pages.page_prospect import page_prospect_body
# from app_pages.page_project_hypothesis import page_project_hypothesis_body
# from app_pages.page_predict_churn import page_predict_churn_body
# from app_pages.page_predict_tenure import page_predict_tenure_body
# from app_pages.page_cluster import page_cluster_body

# Create an instance of the app

# Add your app pages here using .add_page()
app.add_page("The Pipeline", the_project_body)
app.add_page("Data Exploration", data_exploration_page)
app.add_page("Evaluation", evaluation_body)
# app.add_page("Project Summary", project_summary_body)
# app.add_page("Customer Base Churn Study", page_churned_customer_study_body)
# app.add_page("Prospect Churnometer", page_prospect_body)
# app.add_page("Project Hypothesis and Validation",
# page_project_hypothesis_body)
# app.add_page("ML: Prospect Churn", page_predict_churn_body)
# app.add_page("ML: Prospect Tenure", page_predict_tenure_body)
# app.add_page("ML: Cluster Analysis", page_cluster_body)

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
