import streamlit as st
import pandas as pd
from src.data_management import download_files_from_gcs


def data_exploration_page():
    # -- Page title and introduction ---
    st.title("Data Exploration")
    st.write("Explore the dataset to understand its structure and contents.")

    nighttime_wine_with_a_smile = (
        "https://res.cloudinary.com/dybts6jei/image/upload/v1750626557/"
        "phakphoom-srinorajan-hPkUQ30hvRA-unsplash_wtrhvr.jpg"
    )
    st.image(nighttime_wine_with_a_smile)
    st.caption(
        "Photo by "
        "[Phakphoom Srinorajan](https://unsplash.com/photos/"
        "person-holding-wine-glass-with-red-wine-hPkUQ30hvRA)"
    )

    # -- Dataset Overview ---
    st.title("Describing the Dataset")
    st.write("This section provides a brief overview of the dataset used in"
             "this application.")
    df = load_data()
    st.write(df.describe(include="all"))
    st.markdown("""
* The dataset has high cardinality, which may limit the types of libraries and
  algorithms that can be effectively used.

* There is a notable bias toward US wines, which could reduce the appeal of
  recommendations for non-US drinkers. According to a First Leaf study, the US
  leads the global market, consuming 872 million gallons of wine per year—
  representing 19.6% of the global market share. If there model proves a

* The dataset contains duplicate records: while the total count is **280,901**,
  there are only **169,430** unique entries.

* Most wines are rated as very good quality, worth trying, which suggests the
  data is likely to yield strong results for the business requirement:
  **Create a wine recommendation program that identifies wines with similar
  taste profiles based on expert textual descriptions.**

* The wine list is also biased toward Wine Enthusiast reviewers, who are the
  primary end-users for this tool. As a result, most wines fall into the
  **Dinner party hosts, wine-lovers** category.

* The dataset is not balanced, with a significant number of wines rated as
  **very good quality, worth trying**. This could lead to a skewed
  recommendation system that favors these wines.
    """)

    hands_of_grapes = (
        "https://res.cloudinary.com/dybts6jei/image/upload/v1750626553/"
        "maja-petric-vGQ49l9I4EE-unsplash_uuiojw.jpg"
    )
    st.image(hands_of_grapes)

    # -- Dataset Columns Overview ---
    st.title("Dataset Overview")
    st.write("The dataset contains the following columns:")

    df = load_data()
    if df is not None:
        st.dataframe(df.head())
    st.markdown("""
                * **Unnamed: 0**: Numerical numbering of the dataset
                * **country**: Country of origin
                * **description**: Textual description of the wine
                * **points**: Rating points given by Wine Enthusiast
                * **price**: Price of the wine in USD
                * **province**: Province within the country
                * **region_1**: Region within the province
                * **region_2**: Region within the province
                * **variety**: Type of grape used
                * **winery**: Winery that produced the wine
                * **taster_name**: Name of the taster who reviewed the wine
                * **taster_twitter_handle**: Twitter handle of the taster
                * **title**: Name of the wine
                """)

    # -- Dataset Statistics ---
    st.title("Dataset Statistics")
    st.write("Here are some basic statistics about the dataset:")
    st.markdown("""
**Key Highlights:**

* **Dataset Size:** 169,430 records across 14 variables,
  totaling 19.4 MiB in memory.

* **Missing Data:** 21.4% of cells contain missing values,
                representing 507,872 gaps
  that require attention.

* **Variable Types:** The dataset comprises 3 numeric variables and
    11 text-based features, providing a mix of quantitative and descriptive
                data.
    """)

    # --- Missing Values Overview ---
    st.subheader("Missing Values")
    st.markdown(
        "* Due to the dataset consisting of two different studies with the" 
        "later introduction of `taster_name`, `taster_twitter_handle`, " 
        "`title`. Large amount of data is missing."
    )
    # st.dataframe(df.isnull().sum().reset_index(name='missing_count')
    #              .rename(columns={'index': 'column_name'}))

    missingno_matrix = (
        "https://res.cloudinary.com/dybts6jei/image/upload/v1750889632/"
        "missingno_matrix_cdv2jc.png"
    )
    st.image(missingno_matrix)
    st.markdown(
        "The matrix clearly demonstrates where the missing data is situated. "
        "With the lack of information in the top half of the dataset in "
        "regards to the taster, twitter handles and title.\n\n"
        "In early iterations of the project it was suggested to use the "
        "descriptions, points and province to infill the missing data in "
        " region.\n\n"
        "However, with the need to reduce the dataset size, it was decided to "
        "drop all rows with missing values. This decision to reduce "
        "computation time and memory usage, while still retaining a "
        "significant amount of data for analysis."

    )

    # --- Analysis of price and points ---
    st.subheader("Analysis of Price and Points")
    price_df = price_points_analysis(df)
    st.dataframe(price_df)
    st.markdown("""
* This section builds on the initial analysis by providing more detail about
the price, enabling better grouping of the data into different categories.
* All wine prices are listed in US dollars ($).
* The majority of wines are affordable for the average household; even the 99th
  percentile price is attainable at $150.
* The dataset covers a wide price range, from budget-friendly options to
  extreme luxury.
* This indicates that the data is well suited for the target audience of wine
  enthusiasts
""")
    st.subheader("Price Distribution")

    # -- Price Distribution Visualisation ---
    wine_descriptions = (
        "https://res.cloudinary.com/dybts6jei/image/upload/v1750895131/"
        "wine_descriptions_wordcloud_q6wg8d.png"
    )
    st.image(wine_descriptions)
    st.markdown("""
The word cloud above illustrates the most frequently used words in the wine.
It is evident that the descriptions are rich in detail, highlighting the
high cardinality of the dataset. This richness in language is beneficial for
the recommendation system, as it allows for more nuanced comparisons between
wines.
    """)

    # -- Box plot - price and point ---
    boxplot_price_points = (
        "https://res.cloudinary.com/dybts6jei/image/upload/v1750889631/"
        "box_plot_price_by_points_miq1nl.png"
    )
    st.image(boxplot_price_points)
    st.markdown(
        """
As illustrated in the box plot, there is considerably more variance within the
upper price boundaries for higher-rated wines. This observation indicates that
while not all top-rated wines are necessarily the most expensive, the highest
point ratings are associated with a broader and often significantly higher
price range. \n\n

Whereas, wines of middle-range quality can occasionally command prices
exceeding those of exceptional wines, highlighting the complex relationship
between perceived quality and market value.
        """
    )

    # -- Distribution of Points ---

    distribution_points = (
        "https://res.cloudinary.com/dybts6jei/image/upload/v1750889631/"
        "boxplot_points_yac4jg.png"
    )
    st.image(distribution_points)
    st.write(
        "The left-skewed distribution of wine scores indicates that "
        "most wines receive lower ratings, with only a small number "
        "achieving high scores. This is typical in wine ratings, "
        "where exceptional wines are rare. However, the presence "
        "of outliers highlights a few wines with significantly high "
        "ratings. These standouts, while few, can pull the average "
        "score higher than the median, suggesting they're notably "
        "better than the majority. \n\n"
        "This distribution implies that most wines fall within the "
        "very good quality range, offering a solid foundation for "
        "customer satisfaction. The presence of those exceptional, "
        "highly-rated wines creates an aspirational desire for \n\n"
        "end-users, encouraging them to seek out these top-tier selections. "
        "For sommeliers, this means they can confidently meet customer "
        "expectations with a wide range of very good wines, while also "
        "having a few extraordinary options to drive sales and foster "
        "repeat business."
    )

    grape_things = ("https://res.cloudinary.com/dybts6jei/image/"
                    "upload/v1750626541/pexels-grape-things-"
                    "2954924_zh4nel.jpg")

    st.image(grape_things, caption="Photo by [Grape Things]"
             "(https://unsplash.com)")

# --- Key Insights ---
    st.title("Key Insights")
    st.markdown("""
🍷 **Key Conclusions About the Dataset**\n\n
**High Cardinality & Rich Descriptions**\n\n
The dataset has high cardinality,
especially in textual fields like wine descriptions. This richness offers a
strong foundation for a taste-based recommendation system but also increases
computational complexity.

**Bias Toward US Wines and Reviewers**\n\n
There’s a clear geographic and reviewer bias—most wines are US-based and
reviewed by Wine Enthusiast tasters. This could limit relevance for
international users but aligns well with a US-centric audience.

**Quality Ratings Are Skewed**\n\n
The majority of wines are rated as very good quality, creating a skewed
rating distribution. While helpful for identifying strong contenders,
it may lead to a model that over-recommends already well-regarded wines.

**Duplicates and Missing Data**\n\n
Though the dataset contains over 280,000 records, there are only
~169,000 unique entries. Over 21% of cells have missing values,
especially in columns like taster name, Twitter handle, and title.
These were dropped to optimize performance and reduce memory usage.

**Affordability and Aspirational Wines**\n\n
Wines span a broad price range—most are affordable, but there are
a few standout luxury selections. This helps the system cater to
both casual drinkers and connoisseurs seeking rare finds.

**Quality–Price Relationship Is Nonlinear** \n\n
Top-rated wines don’t always have the highest price tags.
Some mid-tier wines are surprisingly expensive, suggesting
brand value or rarity may influence pricing more than objective
quality.
""")
    


# -- Load Data Function ---

def load_data():
    """
    Load the dataset from a CSV file after downloading it from Google Cloud
    Storage if it doesn't exist.
    """
    dest_file = (
        "VineFind_v2/outputs/datasets/collection/wine_reviews_collected.csv",
        dtype={11: str, 12: str, 13: str}
    )
    dtype_dict = {11: str, 12: str, 13: str}
    try:
        download_files_from_gcs(
            bucket_name="vinefind",
            source_blob_name="datasets/collection/wine_reviews_collected.csv",
            destination_file_name=dest_file,
        )
        df = pd.read_csv(dest_file, dtype=)
    except Exception as e:
        st.error(
            f"🍷 Oops! We couldn't fetch the wine dataset needed to "
            f"explore the data. Please try again later. Error: {e}"
        )
        return None

    return df


def price_points_analysis(df):
    """
    Analyze the price and points of the wines in the dataset.
    """
    price = {}
    price['mean'] = df['price'].mean()
    price['median'] = df['price'].median()
    price['mode'] = df['price'].mode()
    price['std'] = df['price'].std()
    price['min'] = df['price'].min()
    price['max'] = df['price'].max()
    price['quantile_25'] = df['price'].quantile(0.25)
    price['quantile_75'] = df['price'].quantile(0.75)
    price['quantile_90'] = df['price'].quantile(0.90)
    price['quantile_95'] = df['price'].quantile(0.95)
    price['quantile_99'] = df['price'].quantile(0.99)

    price_df = pd.DataFrame(price, index=None)
    price_df
    return price_df
