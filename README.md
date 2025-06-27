# ![VineFind Logo](https://res.cloudinary.com/dybts6jei/image/upload/v1750626537/logo_xklz9j.png)

# Welcome

Life’s too short for a bad bottle of wine — and too short to keep grabbing the same one just because it’s familiar. By describing your favourite bottle of wine this predictive analytics tool will help you find your new favourite bottle.

You can visit the live site [here](https://vine-find-b860638a63f0.herokuapp.com/)

## How to use this repo

To set up and run this project, follow these steps. This project uses two branches:  

**final deployment**: the live, production-ready version currently running on Heroku.

 **main**: the editable development branch for ongoing updates and improvements

1. **Repository Setup**: Begin by using this repository as a template to create your own GitHub project repository.

1. **Codespace Creation**: Navigate to your newly created repository on GitHub. Click the green "Code" button and select "Create codespace on main" from the "Codespaces" tab.

1. **Workspace Initialisation**: Allow a few minutes for the Codespace workspace to fully open and initialize.

1. **Dependency Installation**:Install Dependencies: Once the workspace is ready, open a new terminal within Codespaces and run pip install -r requirements.txt to install all necessary project dependencies.

1. **Access Notebooks**: Open the jupyter_notebooks directory and select the desired notebook you wish to work with.

1. **Select Kernel**: Finally, click the kernel button within the notebook interface and choose "Python Environments" to ensure the correct environment is active.

1. This project uses [Git LFS](https://git-lfs.github.com/) for large files. Please run `git lfs install` after cloning.

## Cloud IDE Reminders

To log into the Heroku toolbelt CLI:

1. Log in to your Heroku account and go to _Account Settings_ in the menu under your avatar.
2. Scroll down to the _API Key_ and click _Reveal_
3. Copy the key
4. In the terminal, run `heroku_config`
5. Paste in your API key when asked

You can now use the `heroku` CLI program - try running `heroku apps` to confirm it works. This API key is unique and private to you, so do not share it. If you accidentally make it public, then you can create a new one with _Regenerate API Key_.

## Dataset Content

- The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/zynicide/wine-reviews). The data was scraped from WineEnthusiast during the week of June 15th, 2017 and subsequent data was retrived November 22nd, 2017. 

**Key Highlights about the data:**

* **Dataset Size:** **169,430** records across **14 variables**, totaling 19.4 MiB in memory. After the data preparation stage **14,441** data entries were used to create the model

* **Missing Data:** **21.4%** of cells contain missing values, representing **507,872** gaps that require attention.

* **Variable Types:** The dataset comprises **3 numeric variables** and **11 text-based features**, providing a mix of quantitative and descriptive data.* 

User stories were create based on the roles that often participate within the predictive analytics process. With they main focus being able to select a bottle wine that I would enjoy. 

## Business Requirements

VineFind aims to help users discover wines that precisely match their taste preferences. It achieves this by analysing both user input and expert wine descriptions. The primary goal is to reduce the time and risk associated with selecting new wines, ensuring users are more likely to enjoy their purchases.

The core business requirements driving the development of VineFind are:

* To empower users to discover wines that genuinely match their individual taste profiles.

* To significantly reduce the time and effort users spend on selecting new wines.

* To minimise the risk of dissatisfaction with wine purchases by increasing the likelihood of enjoyment.

* To prevent the experience of "drinking bad wine" by consistently recommending high-quality, relevant options.


## Hypothesis and how to validate?


|Number| Null Hypothesis (H<sub>0</sub>) | Reason for selection | What does it proves? | Testing| Outcome |
|------|---------------------------------|----------------------|----------------------|--------|---------|
|**1.**|The content-based filtering system will not effectively recommend wines based on their taste profile.|This hypothesis will determine whether the overall aim of the project is achieved by recommending wines based on user input.| Rejecting this null would confirm the project’s goal has been met as wine is recommended based on their taste profile.|Focus group feedback and Streamlit **dummy** will be used. Using a likert scale and free text response. |This hypothesis has been rejected. The content-based filtering system was able to have an average score of **72%**. 
  

## The rationale to map the business requirements to the Data Visualisations and ML tasks

- List your business requirements and a rationale to map them to the Data Visualisations and ML tasks.

* To empower users to discover wines that genuinely match their individual taste profiles.

* To significantly reduce the time and effort users spend on selecting new wines.

* To minimise the risk of dissatisfaction with wine purchases by increasing the likelihood of enjoyment.

### To prevent the experience of "drinking bad wine" by consistently recommending high-quality, relevant options.

The dataset is biased towards good quality wines. The distribution in rankings implies that most wines fall within the "very good quality" range, offering a solid foundation for customer satisfaction. The presence of those exceptional, highly-rated wines creates an aspirational desire for end-users, encouraging them to seek out these top-tier selections. For sommeliers, this means they can confidently meet customer expectations with a wide range of very good wines, while also having a few extraordinary options to drive sales and foster repeat business.

The left-skewed distribution of wine scores indicates that most wines receive lower ratings, with only a small number achieving high scores. This is typical in wine ratings, where exceptional wines are rare. However, the presence of outliers highlights a few wines with significantly high ratings. These standouts, while few, can pull the average score higher than the median, suggesting they're notably better than the majority.


![Distrubution of wine scores](https://res.cloudinary.com/dybts6jei/image/upload/v1750889631/boxplot_points_yac4jg.png)

## ML Business Case

- In the previous bullet, you potentially visualised an ML task to answer a business requirement. You should frame the business case using the method we covered in the course.

The system primarily utilises semantic similarity. This involves a pre-trained SentenceTransformer model, using deep learning for natural language processing, to encode both user-provided descriptions of wines they enjoy and professional wine descriptions into numerical representations. Cosine similarity is then applied to match user input to the most relevant wines in the dataset.

Beyond the core recommendation engine, the project also incorporates Exploratory Data Analysis (EDA). This includes using profiling tools, various visualizations (such as boxplots and word clouds), and statistical tests (e.g., chi-squared for price/points association) to understand the dataset.

## Dashboard Design

- List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other items, that your dashboard library supports.
Page 1: The Pipeline
* Heading
* Introduction 
* Simple Tasting and flavors of wine 
* Instructions about how to use the tool
* Example
* Text box
* Button "Get Recommendations"
* Loading message
* feedback form

Side bar 
* Menu
* Logo
* Rating and Price Guide

Page 2: Project Summary
* Aims
* Learning Methods
* Ideal Outcome
* Success and Failure Metric
* Model Output and User Relevance
* Heuristics and Data

Page 3: Data Exploration 
* Describing the datset: Dataframe
* Sample of the dataset: Dataframe
* Feature Strategy: table
* Dataset Statestics
* Missing Values: Misso Matrix
* Analysis of Price and Points
    * Scatterplot
    * Boxplot
* Key insights 

Page 4: Evaluation of Model Performance
* White wine evaluation
* White Wine Evaluation Results
* Analysis of Results
* Top White Wines: Dataframe
* Conclusion
* Key Strengths Observed
* Identified Challenges and Areas for Refinement
* Recommendations for Improvement


## Unfixed Bugs

- The `requirements.txt` file is designed to be comprehensive. However there has been persistent dependency issues. While the exact root cause of these problems remains under investigation, NumPy is suspected to be a frequent contributor.

## Deployment

### Heroku

Our application is currently deployed and accessible live at: [text](https://vine-find-b860638a63f0.herokuapp.com/)

The project was deployed to Heroku, adhering to their guidelines for Python applications. To ensure compatibility, the `.python-version` file was configured to use a Python version supported by the Heroku-20 stack.

The deployment process followed these steps:

1. Set the Python Runtime Ensure the `.python-version` file specifies a Python version supported by the Heroku-20 stack (e.g., python-3.11).

1. Create a New Heroku App Log in to the Heroku dashboard and create a new app with a unique name.

1. Connect to GitHub

1. Navigate to the Deploy tab.

1. Select GitHub as the deployment method.

1. Search for your repository and click Connect.

1. Deploy the App

1. Choose the desired branch (e.g., main).

1. Click Deploy Branch.

1. Once deployment is complete, click Open App to access your live application.

1. Manage Slug Size (if needed) If your app exceeds Heroku's slug size limit, use a 
`.slugignore` file to exclude unnecessary large files from deployment.

Slug Size Management: In instances where the slug size was too large, unnecessary large files were added to the .slugignore file to optimize the deployment.

## Main Data Analysis and Machine Learning Libraries
### Key Insights

🍷 Key Conclusions About the Dataset

### High Cardinality & Rich Descriptions

The dataset has high cardinality, especially in textual fields like wine descriptions. This richness offers a strong foundation for a taste-based recommendation system but also increases computational complexity.

### Bias Toward US Wines and Reviewers

There’s a clear geographic and reviewer bias—most wines are US-based and reviewed by Wine Enthusiast tasters. This could limit relevance for international users but aligns well with a US-centric audience.

### Quality Ratings Are Skewed

The majority of wines are rated as very good quality, creating a skewed rating distribution. While helpful for identifying strong contenders, it may lead to a model that over-recommends already well-regarded wines.

### Duplicates and Missing Data

Though the dataset contains over 280,000 records, there are only ~169,000 unique entries. Over 21% of cells have missing values, especially in columns like taster name, Twitter handle, and title. These were dropped to optimize performance and reduce memory usage.

### Affordability and Aspirational Wines

Wines span a broad price range—most are affordable, but there are a few standout luxury selections. This helps the system cater to both casual drinkers and connoisseurs seeking rare finds.

### Quality–Price Relationship Is Nonlinear

Top-rated wines don’t always have the highest price tags. Some mid-tier wines are surprisingly expensive, suggesting brand value or rarity may influence pricing more than objective quality.

## Main Data Analysis and Machine Learning Libraries

Here, you should list the libraries used in the project and provide an example(s) of how you used these libraries.

- **Pandas** - This library was used for data manipulation to facilate the analysing and presenting data in Dataframes. It allowed for easy handling of tabular data making it to filter and aggreate the data

- **SentenceTransformer** - Used to create multidimensial embeddings enabling the model to learn the semantic meaning 

- **Streamlit** - Provided the platform to develop the dashboard to showcase the pipeline and summarise key findings

- **wordcloud** - Used for data visualisation of textual information to quickly see trends and similarites. This was used both for data understanding at the begining and end of the project

- **Missingno** - An easy to read visualisation to determine the amount of missing data

- **Scipy** - This allowed the use of chi-squared test to create an accurate evaluation whether there were trends in the data, specifically betwen `points` and `price`

- **Matplotlib** - This plotting library allow for the creation of static visuals so the data could be inspected.

- **ydata_profiling** To gain an overview of the data to assess the benefits of using Matplotlib to examine the data.

* **tqdm.notebook** - To estimate the amount of time it takes for the machine learning alorgithm requires

* **joblib** - Used to efficiently save and load files, preserving pandas data types and manipulations. It provides a faster and more robust alternative to pickle for serialising large numpy arrays and pandas objects.

* **Numpy** - Used to create a custom column that allowed the translation of price and points into more user friendly categories. 

* **Scikit-Learn** The cosine similiarity is a module in Scikit-Learn. This formed the second part of the pipeline. Alongside the sentence transformer 


# Credits

### Content

[Gemini](https://gemini.google.com/app) - Used for brainstorming, creations of step by step guides when stuck and explaining documentation. Proofreading all documentation

[copilot](https://copilot.microsoft.com/chats/STrD7LmErWYZQ9zFth7DR) - used to explain code when stuck

[Geekforgeeks - Missingno](https://www.geeksforgeeks.org/machine-learning/python-visualize-missing-values-nan-values-using-missingno-library/)

#### Documentation

* [Pandas](https://pandas.pydata.org/pandas-docs/stable/index.html) 

* [SentenceTransformer](https://huggingface.co/sentence-transformers)

* [Streamlit](https://docs.streamlit.io/)

* [wordcloud- Code Institute tutorials](https://codeinstitute.net/)

* [Scipy](https://scipy.org/)

* [Matplotlib](https://matplotlib.org/)

* [ydata_profiling](https://docs.profiling.ydata.ai/latest/)

* [tqdm.notebook](https://tqdm.github.io/docs/notebook/)

* [joblib](https://joblib.readthedocs.io/en/stable/)

* [Numpy](https://numpy.org/) 

* [Scikit-Learn*](https://scikit-learn.org/stable/index.html)


* Williams, D (2013). A Little course in... Wine Tasting. DK Ltd. London

### Media

[Unsplash](https://unsplash.com/)

[Emoji's](https://emojicopy.com/)

[wine_logo](https://openart.ai/home?msclkid=e70ac5f950151baa6055ada7708506f8&utm_source=bing&utm_medium=cpc&utm_campaign=Ser%20-%20EU%20-%20AI%20generator%20-%20CR%3E1.0%25&utm_term=ai%20image%20generator&utm_content=Top%20Terms%20-%20Exact%20-%200.0)

[Wine Glass Emoji](https://emojiterra.com/wine-glass/)


## Acknowledgements

Thank you to Team Amazing, who always help me at the last minute to do valuable testing.



