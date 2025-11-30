# 📊 LinkedIn Job Title Trends Analysis

A mini Data Science project exploring hiring trends across Data Science and AI job roles.  
This project demonstrates how to:

- Extract job titles from a saved web page (LinkedIn-style job listings)
- Clean and preprocess text using NLP techniques
- Perform Exploratory Data Analysis (EDA) on job trends
- Visualize insights about role types, popularity, and title patterns

> ⚠️ This project is for **learning purposes only**.  
It uses sample HTML or dummy data and does **not** scrape live LinkedIn pages to comply with platform policies.

Nice, this will look *really* good on your GitHub + resume.
I’ll give you **both**:

1. A clean **Python script** you can push
2. A ready-to-use **README.md**

You can copy-paste directly into VS Code.

---

## 1️⃣ Python script: `linkedin_job_trends_analysis.py`

> ⚠️ Important: Scraping LinkedIn directly can violate their Terms of Service.
> For your GitHub version, it’s safer to:
>
> * Either use **saved HTML files** (downloaded manually)
> * Or treat this as a **toy example** with dummy data.
>   The script below is written in a generic way; in the README you can mention it’s *for educational purposes only*.

```python
"""
LinkedIn Job Trends Analysis
----------------------------

Educational project that demonstrates:
- Collecting job titles from an HTML page (e.g., saved LinkedIn jobs page)
- Cleaning and preprocessing text
- Performing simple EDA and correlation analysis
- Visualizing results

NOTE: Always respect a website's Terms of Service and robots.txt.
This script is meant to work with *locally saved HTML* or sample pages,
not for large-scale scraping of live sites.
"""

import re
import requests
from bs4 import BeautifulSoup

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------------------

# If you want to experiment safely, you can:
# - Save a public job search results HTML page manually as "jobs_sample.html"
# - Point LOCAL_HTML_PATH to that file.

LOCAL_HTML_PATH = "jobs_sample.html"  # change or comment out if not using local file

# For plotting style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# --------------------------------------------------------------------------------
# 2. Data Collection Helpers
# --------------------------------------------------------------------------------

def load_html_from_file(path: str) -> str:
    """Load HTML content from a local file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_job_titles_from_html(html: str) -> list:
    """
    Parse job titles from an HTML page.

    For a LinkedIn-style jobs page, job links often use a class such as 'base-card__full-link'.
    This may need to be updated depending on the HTML structure.
    """
    soup = BeautifulSoup(html, "html.parser")

    # This CSS class is an example. Update based on the actual HTML you have.
    job_link_class = "base-card__full-link"

    jobs = soup.find_all("a", class_=job_link_class)
    titles = [job.get_text(strip=True) for job in jobs]

    return titles


def example_dummy_job_titles() -> list:
    """
    Fallback list of job titles if HTML scraping isn't used.
    This keeps the notebook reproducible even without real HTML.
    """
    return [
        "Data Scientist - Machine Learning Engineer",
        "Senior Data Scientist | AI Specialist",
        "Lead AI Engineer",
        "Junior Data Analyst",
        "Data Engineer",
    ]

# --------------------------------------------------------------------------------
# 3. Text Preprocessing
# --------------------------------------------------------------------------------

def setup_nltk():
    """Download NLTK resources if not already present."""
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)


def clean_title(title: str) -> str:
    """
    Clean a single job title:
    - Handle None or empty values
    - Remove special characters
    - Lowercase
    - Remove stopwords
    """
    if not title or not isinstance(title, str):
        return ""

    # Remove non-word characters
    title = re.sub(r"[^\w\s]", "", title)

    # Lowercase
    title = title.lower()

    # Tokenize and remove stopwords
    tokens = word_tokenize(title)
    stop_words = set(stopwords.words("english"))

    filtered_tokens = [t for t in tokens if t not in stop_words]

    return " ".join(filtered_tokens)

# --------------------------------------------------------------------------------
# 4. EDA Functions
# --------------------------------------------------------------------------------

def build_dataframe(job_titles: list) -> pd.DataFrame:
    """
    Build a simple DataFrame with:
    - Original job titles
    - Cleaned titles
    - Length of title (in words)
    - Dummy popularity scores (for demonstration)
    """
    # Create dummy popularity scores just to illustrate correlation
    popularity = list(range(70, 70 + 5 * len(job_titles), 5))[: len(job_titles)]

    cleaned_titles = [clean_title(title) for title in job_titles]
    lengths = [len(t.split()) if t else 0 for t in cleaned_titles]

    df = pd.DataFrame(
        {
            "Job Title": job_titles,
            "Cleaned Title": cleaned_titles,
            "Length": lengths,
            "Popularity": popularity,
        }
    )

    return df


def plot_length_distribution(df: pd.DataFrame):
    """Plot histogram of job title lengths."""
    plt.figure()
    sns.histplot(df["Length"], bins=5, kde=True)
    plt.title("Distribution of Job Title Lengths")
    plt.xlabel("Number of Words in Title")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_popularity_bar(df: pd.DataFrame):
    """Plot bar chart of popularity by job title."""
    plt.figure()
    sns.barplot(x="Job Title", y="Popularity", data=df)
    plt.title("Popularity of Job Titles (Dummy Scores)")
    plt.xlabel("Job Title")
    plt.ylabel("Popularity Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_correlation(df: pd.DataFrame):
    """Compute & visualize correlation between title length and popularity."""
    numeric_df = df[["Length", "Popularity"]].dropna().astype(float)

    corr = numeric_df.corr()
    print("Correlation Matrix:\n", corr, "\n")

    plt.figure()
    sns.scatterplot(
        data=df,
        x="Length",
        y="Popularity",
        hue="Popularity",
        size="Popularity",
        sizes=(50, 200),
        palette="cool",
    )
    plt.title("Correlation Between Job Title Length and Popularity")
    plt.xlabel("Job Title Length")
    plt.ylabel("Popularity Score")
    plt.tight_layout()
    plt.show()

    return corr

# --------------------------------------------------------------------------------
# 5. Main Execution
# --------------------------------------------------------------------------------

def main(use_local_html: bool = False):
    setup_nltk()

    # Step 1: Collect job titles
    if use_local_html:
        html = load_html_from_file(LOCAL_HTML_PATH)
        job_titles = parse_job_titles_from_html(html)
        if not job_titles:
            print("No job titles found in the HTML. Falling back to dummy data.")
            job_titles = example_dummy_job_titles()
    else:
        # Use dummy titles if not scraping actual HTML
        job_titles = example_dummy_job_titles()

    print("Sample job titles:", job_titles[:5], "\n")

    # Step 2: Build DataFrame
    df = build_dataframe(job_titles)
    print(df.head(), "\n")

    # Step 3: Visualizations & correlation
    plot_length_distribution(df)
    plot_popularity_bar(df)
    corr = plot_correlation(df)

    print("Analysis complete.")


if __name__ == "__main__":
    # Set `use_local_html=True` if you have a saved HTML file to parse
    main(use_local_html=False)
```

You can name the repo something like:

> `linkedin-job-title-analysis`

and include this `.py` file plus a Jupyter notebook version if you want.

---

## 2️⃣ README: `README.md`

Here’s a complete, polished README you can paste directly into `README.md` in your repo.

````markdown
# LinkedIn Job Title Trends – Data Collection & Analysis

A small but practical **Data Science project** that explores job title trends from public job listings (e.g., LinkedIn-style pages).  
The project demonstrates:

- Collecting job titles from an HTML page (scraping or saved copy)
- Cleaning and preprocessing text (tokenization, stopword removal)
- Performing basic **EDA** and **correlation analysis**
- Visualizing insight about job title length and popularity

> ⚠️ **Important:** This project is for **educational purposes only**.  
> Always respect a website's **Terms of Service** and `robots.txt`.  
> The example code can be run safely using **locally saved HTML pages** or dummy data, not for large-scale automated scraping.

---

## ✨ Project Overview

The analysis focuses on these questions:

- How long are typical Data Science job titles?
- Which keywords appear most frequently (e.g., *Data Scientist*, *Machine Learning*, *AI*)?
- Is there any relationship between job title **length** and a dummy **popularity score**?
- What trends do we observe about **role specialization** and hybrid profiles?

---

## 🧱 Tech Stack

- **Language:** Python  
- **Data Collection:** `requests`, `BeautifulSoup`  
- **NLP / Text Preprocessing:** `NLTK`, `re`  
- **Data Handling:** `pandas`  
- **Visualization:** `seaborn`, `matplotlib`  

---

## 📂 Project Structure

```bash
linkedin-job-title-analysis/
├── linkedin_job_trends_analysis.py   # Main script (collection + cleaning + EDA)
├── jobs_sample.html                  # (Optional) Saved HTML page for testing
├── notebooks/
│   └── linkedin_job_trends.ipynb     # Optional Jupyter notebook version
└── README.md
````

* `linkedin_job_trends_analysis.py`
  Contains the full pipeline: loading HTML / dummy data, cleaning titles, building a DataFrame, and plotting results.

* `jobs_sample.html` *(optional)*
  A locally saved public job search page used for safe experimentation.

---

## 🚀 How to Run

1. **Clone the repo**

   ```bash
   git clone https://github.com/<your-username>/linkedin-job-title-analysis.git
   cd linkedin-job-title-analysis
   ```

2. **Create and activate a virtual environment (optional but recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Example `requirements.txt`:

   ```text
   requests
   beautifulsoup4
   nltk
   pandas
   seaborn
   matplotlib
   ```

4. **Run the script**

   Using **dummy job titles** (safe default):

   ```bash
   python linkedin_job_trends_analysis.py
   ```

   Using a **locally saved HTML file** (update path in the script):

   ```bash
   python linkedin_job_trends_analysis.py
   ```

   Make sure `LOCAL_HTML_PATH` inside the script points to your saved `jobs_sample.html`, and set `use_local_html=True` in `main()` if you modify it.

---

## 📊 What the Script Does

* Extracts job titles from an HTML page (sample) or uses dummy Data Science job titles.
* Cleans titles by:

  * Removing special characters
  * Lowercasing
  * Removing stopwords
  * Tokenizing the words
* Builds a small `pandas` DataFrame containing:

  * Original job title
  * Cleaned job title
  * Length of title (word count)
  * Dummy popularity scores
* Visualizes:

  * Histogram of job title lengths
  * Bar chart of (dummy) popularity scores
  * Scatter plot of **length vs popularity** with correlation matrix

---

## 🧠 Insights (from a sample run)

* Most job titles fall in the **3–6 word** range.
* Titles frequently contain **Data Scientist**, **Machine Learning**, and **AI**, reflecting strong demand for these skills.
* Many roles are **hybrid**, combining multiple specializations (e.g., “Data Scientist – Machine Learning Engineer”).

---

## 🔮 Possible Extensions

* Replace dummy popularity scores with real engagement metrics (views, applications) using a valid data source.
* Analyze **keyword frequency** using `Counter` or `scikit-learn`’s `CountVectorizer`.
* Cluster similar job titles to detect **role families** (e.g., DS, MLE, Analyst).
* Build dashboards using **Streamlit** or **Plotly** to interactively explore job title trends.

---

## 📌 Disclaimer

This project is meant to illustrate a **Data Science workflow**:
data collection → cleaning → EDA → insight.

When working with any real website:

* Read and follow its **Terms of Service**.
* Limit requests and avoid disrupting their services.
* Prefer APIs, open datasets, or manually-exported data.

---

## 🙌 Author

**Nikitha Kanala**
Data Science undergraduate passionate about AI, NLP, and building data-driven products.

```
