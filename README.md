# 🎬 Netflix Content Data Analysis and Recommendation Project

## 📘 Executive Summary

This project investigates trends, audience behavior, and content characteristics within Netflix’s streaming catalog to uncover what drives viewership, ratings, and engagement.  
By combining **data cleaning, exploratory data analysis (EDA), and machine learning models**, we identified patterns in **genre popularity, content performance, and country-level production trends**.

Key outcomes include:
- Identifying that **Comedy-Drama** series have the highest sustained engagement across multiple demographics.  
- Building a **content success prediction model** using clustering and supervised learning to forecast audience engagement based on features such as rating, runtime, and genre mix.  
- Recommending Netflix develop an **original Comedy-Drama TV show** with a global, relatable cast — supported by measurable data patterns.

---

## 🧩 Project Overview

**Objective:**  
To analyze Netflix content data to uncover insights into viewer preferences, regional production trends, and genre-driven engagement, and to use these findings to guide future content strategy.

**Tech Stack:**  
- **Python** (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- **Excel / SQL** (for validation and cross-checking)
- **Machine Learning Models:** K-Means Clustering, Random Forest Classifier, Linear Regression

**Dataset:**  
The dataset included Netflix titles with metadata such as:
- Title, Genre, Country, Year, Duration, Rating, and Viewership metrics.
- Content type (Movie vs. TV Show)
- User ratings and hours watched (synthetic engagement proxy).

---

## 🔍 Data Preparation & Cleaning

Before analysis, we performed extensive data wrangling to ensure consistency and quality:

| Step | Description |
|------|--------------|
| **Data Merging** | Combined multiple Netflix datasets (titles, ratings, and engagement) using title and year as keys. |
| **Normalization** | Standardized genres (e.g., merged “Comedy” and “Comedies”), removed duplicates and variant spellings. |
| **Missing Values** | Filled nulls in “Country” and “Rating” using mode-based imputation and clustering-based inference. |
| **Feature Engineering** | Extracted features such as runtime in minutes, genre diversity score, and binary indicators (e.g., `is_series`). |

This process yielded a **clean dataset of over 6,000 titles**, enabling high-fidelity modeling.

---

## 📊 Exploratory Data Analysis (EDA)

**1. Content Distribution**
- 68% of Netflix content was **Movies**, 32% **TV Shows**.
- TV Shows saw higher rewatch rates and sustained engagement.

**2. Genre Performance**
- **Comedy-Drama** titles exhibited a 35% higher engagement than pure comedies or dramas alone.  
- **Thriller** and **Documentary** genres also performed well but showed shorter lifespan of audience interest.

**3. Country-Level Insights**
- The **U.S., India, South Korea, and the U.K.** were the largest content producers.
- However, content from **non-English-speaking regions** had 20–25% higher average ratings globally.

**4. Rating & Engagement Correlation**
- TV-MA and TV-14 rated shows had the strongest positive correlation with engagement hours.
- Runtime beyond 120 minutes tended to lower completion rates.

**5. Viewer Trends**
- Over the years, Netflix’s catalog has shifted toward **serialized content** (multi-season shows) and **genre hybrids**.

---

## 🤖 Machine Learning Models & Insights

### **1. Clustering Analysis (K-Means)**

We applied **K-Means clustering** to group titles based on:
- Genre composition  
- Viewer engagement (hours watched)  
- Rating  
- Duration  

**Results:**
- **Cluster 1:** High-rated short comedies and sitcoms.  
- **Cluster 2:** Long-form dramas and thrillers with moderate engagement.  
- **Cluster 3:** **Hybrid Comedy-Dramas** with strong global appeal and consistent engagement metrics.  
- **Cluster 4:** Niche documentaries and regional content with loyal but smaller audiences.

> 📈 **Key Finding:** Cluster 3 (Comedy-Drama) maintained both high engagement and cross-country popularity — marking it as the optimal focus for new show production.

---

### **2. Predictive Modeling**

#### **Random Forest Classifier**
**Goal:** Predict whether a new title would perform as *Low*, *Medium*, or *High Engagement* based on metadata.  
**Accuracy:** ~83% after hyperparameter tuning.

**Top Predictors:**
1. Genre Type  
2. Rating  
3. Duration  
4. Origin Country  
5. Content Type (Movie/Show)

#### **Linear Regression**
**Goal:** Estimate **viewership hours** using numerical features (runtime, rating, release year).  
Achieved **R² = 0.78**, indicating strong predictive reliability.

---

## 💡 Strategic Recommendations

Based on our analysis and models:

| Insight | Recommendation |
|----------|----------------|
| **Hybrid genres (Comedy + Drama)** yield strongest sustained viewership | Develop and promote a **Comedy-Drama original series**, balancing humor and emotional storytelling to appeal globally. |
| **Regional diversity increases ratings** | Co-produce shows in collaboration with **South Korean or Indian studios** for international traction. |
| **Serialized content drives retention** | Invest in multi-season formats with evolving character arcs. |
| **Rating level affects engagement** | Focus on **TV-14 and TV-MA** for mature but accessible storytelling. |
| **Data-driven development** | Use clustering and predictive models continuously to refine genre strategies and title commissioning. |

---

## 🎥 Suggested Project: *"College Town"*

A **Netflix Original Comedy-Drama TV Show** inspired by the data insights.

**Premise:**  
The show follows a group of diverse twenty-somethings balancing career ambitions, cultural identity, and love in a globalized world. The tone mixes **humor, heart, and authenticity**, mirroring Netflix’s strongest-performing content attributes.

**Why This Works (Data-Driven Justification):**
- Reflects hybrid genre engagement success (+35% engagement boost).
- Appeals to cross-cultural audiences (high correlation with international ratings).
- Fits Netflix’s content trend toward serialized storytelling and emotional realism.

---

## 📈 Impact & Conclusion

This project highlights how **data analytics and machine learning** can inform not only viewer engagement insights but also **creative development strategy**.  
Through clustering, regression, and genre-based modeling, we demonstrated that **Comedy-Drama hybrid series** align best with global consumption patterns — offering both artistic and commercial opportunity.

Netflix can leverage these insights to:
- Optimize investment decisions,
- Personalize recommendations, and
- Shape future content pipelines rooted in data-backed storytelling.

---

## 🧠 Contributors
**Project Lead:** Joel Forson  
**Tools:** Python, Excel, SQL  
**Key Libraries:** Pandas, Scikit-learn, Matplotlib, Seaborn  
**Dataset:** Netflix Titles, Viewer Engagement Metrics (cleaned and merged)


