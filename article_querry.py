# article_query.py

import pandas as pd

def load_data(path="raw_articles.csv"):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print("❌ raw_articles.csv not found. Run --fetch-articles first.")
        return pd.DataFrame()

def filter_articles(df, term=None, institution=None, year=None, keyword=None, limit=10):
    if term:
        term = term.lower()
        df = df[df['title'].str.lower().str.contains(term, na=False) | 
                df['abstract'].str.lower().str.contains(term, na=False)]

    if institution:
        df = df[df['institutions'].str.lower().str.contains(institution.lower(), na=False)]

    if year:
        df = df[df['year'].astype(str) == str(year)]

    if keyword:
        df = df[df['keywords'].str.lower().str.contains(keyword.lower(), na=False)]

    return df.head(limit)