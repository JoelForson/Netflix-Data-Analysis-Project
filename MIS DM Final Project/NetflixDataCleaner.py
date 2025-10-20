import pandas as pd
import numpy as np
import re

df_netflix = pd.read_csv("/Users/joelforson/Downloads/Merged_Netflix_Dataset-1.csv")
df_titles = pd.read_csv("/Users/joelforson/Downloads/titles.csv")


# Select/Retain columns & Inital Cleaning

columns_to_keep = ["title", "Available Globally?", "Release Date", "Hours Viewed", "Runtime", "Views"]
df_netflix = df_netflix[columns_to_keep]


# Convert "Runtime" into Minutes

def safe_runtime_to_minutes(runtime_value):
    """
    Converts a 'HH:MM' string to total minutes as an integer.
    If it's numeric (e.g. '1214.0'), tries float.
    Returns NaN if conversion fails.
    """
    if pd.isnull(runtime_value):
        return np.nan
    
    runtime_str = str(runtime_value).strip()
    if ":" in runtime_str:
        try:
            hours, minutes = map(int, runtime_str.split(":"))
            return hours * 60 + minutes
        except:
            return np.nan
    else:
        # If not 'HH:MM', assume it's numeric minutes
        try:
            return float(runtime_str)
        except:
            return np.nan

df_netflix["Runtime_minutes"] = df_netflix["Runtime"].apply(safe_runtime_to_minutes)
mean_runtime = df_netflix["Runtime_minutes"].mean()
df_netflix["Runtime_minutes"].fillna(mean_runtime, inplace=True)


# Standardize Titles to Merge

def standardize_title(title_str):
    """
    - Removes season references (": season 2", etc.)
    - Removes (YYYY) in parentheses
    - Splits off anything after // or :
    - Removes punctuation
    - Lowercases & strips
    """
    if pd.isnull(title_str):
        return ""
    title_str = str(title_str)
    
    # Remove references like ": season 2", etc.
    title_str = re.sub(r":\s*(season|limited series|series|book|temporada|sezon|part|saison)\s*\d*", "", title_str, flags=re.IGNORECASE)
    # Remove year parentheses: e.g. (2011)
    title_str = re.sub(r"\(\d{4}\)", "", title_str)
    # Split off subtitles after // or :
    title_str = re.split(r'//|:', title_str)[0]
    # Remove punctuation
    title_str = re.sub(r'[^\w\s]', '', title_str)
    return title_str.strip().lower()

df_netflix["standard_title"] = df_netflix["title"].apply(standardize_title)
df_titles["standard_title"] = df_titles["title"].fillna("").apply(standardize_title)


# Merge the Datasets by the title

df_merged = pd.merge(
    df_netflix,
    df_titles,
    on="standard_title",
    how="inner",
    suffixes=("_netflix", "_titles")
)

# 6. Restore original Titles to shows

release_year_col = "release_year" if "release_year" in df_merged.columns else None

if release_year_col:
    # Identify titles with multiple distinct release years
    title_year_counts = df_merged.groupby("standard_title")[release_year_col].nunique()
    multiyear_titles = title_year_counts[title_year_counts > 1].index
    
    def restore_original_title(row):
        """
        If this standard_title has multiple release years,
        append the year to the original 'title_titles' to differentiate.
        Otherwise, just keep 'title_titles'.
        """
        if row["standard_title"] in multiyear_titles:
            yr = row[release_year_col]
            if pd.notnull(yr):
                return f"{row['title_titles']} ({int(yr)})"
        return row["title_titles"]
    
    df_merged["finalized_title"] = df_merged.apply(restore_original_title, axis=1)
else:
    # If no release_year, just keep the original title
    df_merged["finalized_title"] = df_merged["title_titles"]


# Drop Duplicates Based on Release Date, Hours Viewed, Views

# Keep the first occurrence so one original row remains
df_merged.drop_duplicates(subset=["standard_title","Hours Viewed", "Views"], keep="first", inplace=True)
df_merged.drop_duplicates(subset=["finalized_title"], keep="first", inplace=True)

# 8. Clean up and Reorder Columns

to_drop = ["title_netflix", "title_titles", "Runtime"]
for col in to_drop:
    if col in df_merged.columns:
        df_merged.drop(columns=col, inplace=True)

main_cols = [
    "finalized_title",
    "standard_title",
    "Available Globally?",
    "Release Date",
    "Hours Viewed",
    "Runtime_minutes",
    "Views",
]

# Add extras if present
extras = ["release_year", "type", "genres", "production_countries", "imdb_score", "imdb_votes", "tmdb_popularity", "tmdb_score"]
for c in extras:
    if c in df_merged.columns and c not in main_cols:
        main_cols.append(c)

# Reorder columns
main_cols = [c for c in main_cols if c in df_merged.columns]
df_final = df_merged[main_cols + [c for c in df_merged.columns if c not in main_cols]]


# FInal Cleanup for unnoticed errors

#Duplicate Rows?
df_final.duplicated()

#drop unnessecary columns 
df_final = df_final.drop(columns=['imdb_id','id'])

#filling in nulls
mode_imdb = df_final['imdb_score'].mode()
mode_imdb[0]
mode_imdb2 = df_final['imdb_votes'].mode()
mode_imdb[0]
mode_tmdb = df_final['tmdb_popularity'].mode()
mode_tmdb[0]
mode_tmdb2 = df_final['imdb_score'].mode()
mode_tmdb[0]

df_final['seasons'].fillna('1', inplace=True)
df_final['age_certification'].fillna('Unkown', inplace=True)
df_final['imdb_score'].fillna(mode_imdb[0], inplace=True)
df_final['imdb_votes'].fillna(mode_imdb2[0], inplace=True)
df_final['tmdb_popularity'].fillna(mode_tmdb[0], inplace=True)
df_final['tmdb_score'].fillna(mode_tmdb2[0], inplace=True)
null = df_final.isnull().sum()


#Remove brackets and quotes

def remove_brackets_and_quotes(value):
    if isinstance(value, str):
        # Remove [ and ], and "
        return value.replace('[','').replace(']','').replace("'",'')
    else:
        return value

for col in df_final.columns:
    if df_final[col].dtype == object:
        df_final[col] = df_final[col].apply(remove_brackets_and_quotes)



# SAVE THE FINAL DATASET!!!

df_final.to_csv("Finalized_Netflix_Dataset_.csv", index=True)
