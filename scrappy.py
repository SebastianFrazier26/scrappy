# taxonomy_with_wordcloud.py (CLI version with IEEE API, persistent tracking, dynamic terms)

# taxonomy_with_wordcloud.py (CLI version with IEEE API, persistent tracking, dynamic terms)

import argparse
import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import nltk
import time
import json
from datetime import datetime
import os
from article_querry import load_data, filter_articles

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def load_classifier():
    with open("classifier_data.json", "r") as f:
        training_data = json.load(f)
    texts, labels = zip(*training_data)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    X_train = vectorizer.fit_transform(texts)
    model = LogisticRegression()
    model.fit(X_train, labels)
    return model, vectorizer

def predict_term_category(term, model, vectorizer):
    X = vectorizer.transform([term])
    return model.predict(X)[0]

TERMS_FILE = "search_terms.json"

### === SEARCH TERM MANAGEMENT === ###
def get_search_terms():
    if os.path.exists(TERMS_FILE):
        with open(TERMS_FILE, 'r') as f:
            base_terms = json.load(f)
    else:
        base_terms = [
            "healthcare cyber security", "hospital cyber security",
            "hospital safety", "healthcare safety",
            "hospital trustworthiness", "public health cyber security"
        ]

    print("\nCurrent search terms:")
    for t in base_terms:
        print(f"- {t}")

    add_more = input("\nWould you like to add more search terms manually? (y/n): ").strip().lower()
    if add_more == 'y':
        print("Enter new terms one per line. Type 'done' when finished:")
        while True:
            new_term = input("New term: ").strip()
            if new_term.lower() == 'done':
                break
            if new_term and new_term not in base_terms:
                base_terms.append(new_term)

    with open(TERMS_FILE, 'w') as f:
        json.dump(base_terms, f, indent=2)

    return base_terms

### === DATA SOURCES === ###
def fetch_ieee(query, api_key, max_records=200):
    base_url = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
    all_results = []

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for term in query:
        params = {
            "apikey": api_key,
            "querytext": term,
            "max_records": max_records,
            "start_record": 1
        }

        try:
            res = requests.get(base_url, params=params, headers=headers, timeout=10)
            res.raise_for_status()
            data = res.json()
        except requests.exceptions.RequestException as e:
            print(f"[❌] IEEE API error for term '{term}': {e}")
            continue
        except ValueError:
            print(f"[❌] Failed to parse JSON for IEEE term '{term}'. Raw response:\n{res.text}")
            continue

        results = data.get("articles", [])
        if not results:
            print(f"[⚠️] No articles returned for IEEE term: {term}")

        for article in results:
            all_results.append({
                "source": "IEEE",
                "title": article.get("title"),
                "abstract": article.get("abstract"),
                "url": article.get("html_url"),
                "authors": [a.get("full_name") for a in article.get("authors", [])],
                "institutions": [],
                "keywords": article.get("index_terms", {}).get("ieee_terms", {}).get("terms", []),
                "term": term,
                "year": article.get("publication_year", "")
            })

        time.sleep(1)

    return all_results


def fetch_google_scholar(query, serpapi_key, limit=200):
    all_results = []

    for term in query:
        params = {
            "engine": "google_scholar",
            "q": term,
            "api_key": serpapi_key
        }
        res = requests.get("https://serpapi.com/search", params=params)
        items = res.json().get("organic_results", [])[:limit]

        for item in items:
            all_results.append({
                "source": "GoogleScholar",
                "title": item.get("title"),
                "abstract": item.get("snippet"),
                "url": item.get("link"),
                "authors": [],
                "institutions": [],
                "keywords": "",
                "term": term
            })
        time.sleep(1)
    return all_results

# (Other source fetch functions like Semantic Scholar, PubMed, arXiv, Google Scholar should be here)
def fetch_semantic_scholar(query, limit=100):
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    all_results = []

    headers = {
        "x-api-key": SEMANTIC_SCHOLAR_API_KEY,
        "User-Agent": "Mozilla/5.0"
    }

    for term in query:
        print(f"[Semantic Scholar] Querying: {term}")
        params = {
            "query": term,
            "limit": limit,
            "fields": "title,abstract,year,authors,url"
        }

        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"[❌] Semantic Scholar API error for '{term}': {e}")
            continue
        except ValueError:
            print(f"[❌] Failed to decode JSON for term '{term}'. Raw response:\n{response.text}")
            continue

        results = data.get("data", [])
        if not results:
            print(f"[⚠️] No results returned for: {term}")

        for paper in results:
            authors = paper.get("authors", [])
            all_results.append({
                "source": "SemanticScholar",
                "title": paper.get("title"),
                "abstract": paper.get("abstract"),
                "url": paper.get("url"),
                "authors": [a['name'] for a in authors],
                "institutions": [a.get('affiliations', []) for a in authors],
                "keywords": "",
                "term": term,
                "year": paper.get("year", "")
            })

        time.sleep(1)  # Enforce 1 request per second (API limit)

    return all_results

def fetch_pubmed(query, retmax=200):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    all_results = []

    for term in query:
        try:
            search = requests.get(f"{base_url}esearch.fcgi", params={
                "db": "pubmed",
                "term": term,
                "retmode": "json",
                "retmax": retmax
            }).json()
            ids = search.get("esearchresult", {}).get("idlist", [])

            fetch = requests.get(f"{base_url}efetch.fcgi", params={
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "xml"
            })
            soup = BeautifulSoup(fetch.content, "xml")
        except Exception as e:
            print(f"PubMed fetch error for '{term}': {e}")
            continue

        for article in soup.find_all("PubmedArticle"):
            title = article.ArticleTitle.text if article.ArticleTitle else ""
            abstract = article.find("AbstractText").text if article.find("AbstractText") else ""
            affs = article.find_all("AffiliationInfo")
            institutions = [aff.Affiliation.text for aff in affs if aff.Affiliation]

            all_results.append({
                "source": "PubMed",
                "title": title,
                "abstract": abstract,
                "url": "",
                "authors": [],
                "institutions": institutions,
                "keywords": "",
                "term": term,
                "year": ""
            })
        time.sleep(1)
    return all_results


def fetch_arxiv(query, max_results=200):
    import xml.etree.ElementTree as ET
    all_results = []

    for term in query:
        try:
            url = f"http://export.arxiv.org/api/query?search_query=all:{term.replace(' ', '+')}&start=0&max_results={max_results}"
            res = requests.get(url)
            res.raise_for_status()
            root = ET.fromstring(res.content)
        except Exception as e:
            print(f"arXiv API error for '{term}': {e}")
            continue

        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text
            abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text
            authors = [a.find("{http://www.w3.org/2005/Atom}name").text for a in entry.findall("{http://www.w3.org/2005/Atom}author")]
            link = entry.find("{http://www.w3.org/2005/Atom}id").text

            all_results.append({
                "source": "arXiv",
                "title": title.strip(),
                "abstract": abstract.strip(),
                "url": link,
                "authors": authors,
                "institutions": [],
                "keywords": "",
                "term": term,
                "year": ""
            })
        time.sleep(1)
    return all_results

### === COLLECTOR === ###
def collect_all_articles(search_terms, serpapi_key, ieee_api_key):
    print("\nCollecting articles from all sources...\n")
    articles = []

    print("Fetching from IEEE Xplore...")
    ieee_articles = fetch_ieee(search_terms, ieee_api_key)
    articles += tqdm(ieee_articles, desc="IEEE")

    print("Fetching from Semantic Scholar...")
    sem_scholar_articles = fetch_semantic_scholar(search_terms, limit=200)
    articles += tqdm(sem_scholar_articles, desc="Semantic Scholar")

    print("Fetching from PubMed...")
    pubmed_articles = fetch_pubmed(search_terms, retmax=200)
    articles += tqdm(pubmed_articles, desc="PubMed")

    print("Fetching from arXiv...")
    arxiv_articles = fetch_arxiv(search_terms, max_results=200)
    articles += tqdm(arxiv_articles, desc="arXiv")

    print("Fetching from Google Scholar...")
    scholar_articles = fetch_google_scholar(search_terms, serpapi_key, limit=200)
    articles += tqdm(scholar_articles, desc="Google Scholar")

    if not articles:
        print("⚠️  No articles found from any source.")
        return pd.DataFrame()

    df = pd.DataFrame(articles)

    if 'institutions' not in df.columns:
        df['institutions'] = ""

    df.drop_duplicates(subset="title", inplace=True)

    df['institutions'] = df['institutions'].apply(
        lambda x: '; '.join(i for sub in x for i in (sub if isinstance(sub, list) else [sub])) if isinstance(x, list) else ''
    )

    return df

### === PERSISTENCE === ###
def merge_with_existing(df_new):
    if df_new is None or df_new.empty or 'title' not in df_new.columns:
        print("No new articles found to merge.")
        return pd.DataFrame()

    today = datetime.now().strftime('%Y-%m-%d')
    df_new['date_added'] = today
    if os.path.exists("raw_articles.csv"):
        df_existing = pd.read_csv("raw_articles.csv")
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.drop_duplicates(subset="title", keep="first", inplace=True)
    else:
        df_combined = df_new

    new_titles = set(df_new['title'])
    existing_titles = set(df_combined[df_combined['date_added'] != today]['title'])
    newly_added = [row for _, row in df_combined.iterrows() if row['title'] in new_titles and row['title'] not in existing_titles]

    if newly_added:
        print("\nNew Articles Added:")
        for row in newly_added:
            print(f"\nTitle: {row['title']}\nYear: {row.get('year', 'Unknown')}\nAdded on: {today}\n-")
    else:
        print("\nNo new articles added this run.")

    df_combined.to_csv("raw_articles.csv", index=False)
    return df_combined

### === TOPIC SUGGESTIONS === ###
def suggest_terms_from_articles(df, top_n=100):
    from collections import Counter

    # Expansion strategies
    THEMES_INFRA = ["security", "safety", "trustworthiness"]
    THEMES_THREAT = ["hospitals", "healthcare", "public health"]
    THREAT_KEYWORDS = {"ransomware", "malware", "attack", "breach", "incident", "phishing", "exploit", "intrusion"}

    # Load existing terms
    if os.path.exists(TERMS_FILE):
        with open(TERMS_FILE, "r") as f:
            existing_terms = set(t.strip().lower() for t in json.load(f))
    else:
        existing_terms = set()

    # Extract term frequencies from combined title+abstract
    text_data = (df['title'].fillna('') + " " + df['abstract'].fillna('')).tolist()
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 3), max_features=1000)
    X = vectorizer.fit_transform(text_data)
    freqs = X.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, freqs), key=lambda x: -x[1])

    # Filter and select top terms
    suggestions = [(term, int(freq)) for term, freq in ranked if term.lower() not in existing_terms][:top_n]

    # Context examples
    context_map = {}
    for term, _ in suggestions:
        term_lower = term.lower()
        matches = [txt for txt in text_data if term_lower in txt.lower()]
        context_map[term] = matches[:2]

    print("\nSuggested additional search terms based on article content:\n")
    for term, freq in suggestions:
        print(f"🔹 {term} (appeared {freq} times)")
        for i, ex in enumerate(context_map[term]):
            short = (ex[:100] + '...') if len(ex) > 100 else ex
            print(f"    Example {i+1}: {short}")
        print()

    # Ask user to add contextualized terms
    auto_add = input("Would you like to add contextualized versions of these terms to your saved search terms? (y/n): ").strip().lower()
    if auto_add == 'y':
        new_terms = []
        health_keywords = {"hospital", "health", "healthcare", "medical", "clinic", "public health"}
        for term, _ in suggestions:
            term_lower = term.lower()
            tokens = set(term_lower.split())

            # Choose strategy based on keyword content
            model, vectorizer = load_classifier()
            new_terms = []

            for term, _ in suggestions:
                category = predict_term_category(term, model, vectorizer)

            if category == "threat":
                for theme in THEMES_THREAT:
                    contextual_term = f"{term} AND {theme}"
                    if contextual_term.lower() not in existing_terms:
                        tokens = set(contextual_term.lower().split())
                        if not any(h in tokens for h in health_keywords):
                            contextual_term += " AND healthcare"
                        new_terms.append(contextual_term)
            else:
                for theme in THEMES_INFRA:
                    contextual_term = f"{term} AND {theme}"
                    if contextual_term.lower() not in existing_terms:
                        if not any(h in tokens for h in health_keywords):
                            contextual_term += " AND healthcare"
                        new_terms.append(contextual_term)

        # Update terms file
        with open(TERMS_FILE, 'r') as f:
            current_terms = json.load(f)
        current_terms.extend(new_terms)
        with open(TERMS_FILE, 'w') as f:
            json.dump(current_terms, f, indent=2)

        print(f"\n✅ Added {len(new_terms)} new contextualized search terms:")
        for t in new_terms:
            print(f"  - {t}")


def recursive_keyword_tree(base_term, api_source, depth, visited=None):
    if visited is None:
        visited = set()

    tree = {}

    # Prevent loops and duplication
    if base_term in visited or depth <= 0:
        return tree

    visited.add(base_term)

    # Fetch articles
    print(f"\n[Depth {depth}] Exploring: \"{base_term}\"")
    if api_source == "semantic_scholar":
        articles = fetch_semantic_scholar([base_term], limit=50)
    elif api_source == "pubmed":
        articles = fetch_pubmed([base_term], retmax=50)
    elif api_source == "arxiv":
        articles = fetch_arxiv([base_term], max_results=50)
    else:
        print("Unsupported API for tree building.")
        return {}

    if not articles:
        return {}

    # Turn into DataFrame
    df = pd.DataFrame(articles)
    text_data = (df['title'].fillna('') + " " + df['abstract'].fillna('')).tolist()

    # Extract top N keywords
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=500)
    X = vectorizer.fit_transform(text_data)
    freqs = X.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, freqs), key=lambda x: -x[1])

    children = [term for term, _ in ranked[:5] if term not in visited and len(term.split()) <= 3]

    for child in children:
        base_words = set(base_term.lower().split())
        child_words = [w for w in child.lower().split() if w not in base_words]
        if not child_words:
            continue  # skip if child is fully redundant
        combined = base_term + " " + " ".join(child_words)
        tree[child] = recursive_keyword_tree(combined, api_source, depth - 1, visited)
    return tree

### === TAXONOMY === ###
def build_taxonomy(df):
    texts = (df["title"].fillna('') + " " + df["abstract"].fillna('')).astype(str).tolist()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(texts, show_progress_bar=True)
    topic_model = BERTopic(verbose=True, n_gram_range=(1, 5))
    topics, probs = topic_model.fit_transform(texts, embeddings)
    df['Topic'] = topics
    df['topic_keywords'] = df['Topic'].apply(
        lambda t: ', '.join([kw for kw, _ in topic_model.get_topic(t)[:5]]) if t != -1 else 'No topic')
    return df, topic_model

def generate_wordcloud(topic_model, df):
    all_words = {}
    for index, row in df.iterrows():
        topic_id = row['Topic']
        if topic_id == -1:
            continue
        topic_words = topic_model.get_topic(topic_id)
        for word, weight in topic_words:
            if any(stop in word.lower().split() for stop in STOPWORDS):
                continue
            all_words[word] = all_words.get(word, 0) + weight

    wc = WordCloud(width=1600, height=800, background_color='white')
    wordcloud = wc.generate_from_frequencies(all_words)
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("wordcloud.png")
    plt.show()

### === MAIN === ###
def main(args):
    search_terms = get_search_terms()
    if args.build_tree:
        print(f"\nStarting recursive topic tree with API: {args.tree_api}, depth: {args.depth}")
        search_terms = get_search_terms()
        tree = {}

        for base in search_terms:
                tree[base] = recursive_keyword_tree(base, args.tree_api, args.depth)

        with open("topic_tree.json", "w") as f:
                json.dump(tree, f, indent=2)

        print("\nRecursive topic tree saved to topic_tree.json")
        return

    if args.fetch_articles:
        search_terms = get_search_terms()
        df_new = collect_all_articles(search_terms, SERPAPI_KEY, IEEE_API_KEY)
        if not df_new.empty:
            merge_with_existing(df_new)
        else:
            print("No new articles fetched.")

    if args.suggest_terms:
        suggested_terms = suggest_terms_from_articles(df_new)
        auto_add = input("\nWould you like to add any of these suggested terms to your saved search terms? (y/n): ").strip().lower()
        if auto_add == 'y':
            with open(TERMS_FILE, 'r') as f:
                current_terms = json.load(f)
            for term in suggested_terms:
                if term not in current_terms:
                    current_terms.append(term)
            with open(TERMS_FILE, 'w') as f:
                json.dump(current_terms, f, indent=2)
            print("Search terms updated with suggestions.")

    if args.query_articles:
        df = load_data()
        results = filter_articles(
            df,
            term=args.term,
            institution=args.institution,
            year=args.year,
            keyword=args.keyword,
            limit=args.limit
        )
        if results.empty:
            print("No articles matched your criteria.")
        else:
            for _, row in results.iterrows():
                print(f"Title: {row['title']}")
                print(f"Year: {row.get('year', 'N/A')}")
                print(f"Institutions: {row.get('institutions', '')}")
                print(f"URL: {row.get('url', '')}")
                print("-" * 60)

    if args.build_taxonomy:
        if not os.path.exists("raw_articles.csv"):
            print("❌ Cannot build taxonomy: raw_articles.csv does not exist. Run --fetch-articles first.")
            return

        df_all = pd.read_csv("raw_articles.csv")
        df_with_topics, model = build_taxonomy(df_all)
        df_with_topics[['title', 'institutions', 'topic_keywords']].to_csv("taxonomy_output.csv", index=False)
        generate_wordcloud(model, df_with_topics)
        print("✅ Done. Outputs saved to taxonomy_output.csv, wordcloud.png, and word_article_map.json")

# TODO: REPLACE WITH YOUR API KEYS - example keys provided
if __name__ == "__main__":
    SERPAPI_KEY = "d5addf0b54ca630f94057f17f33ea25becbd0586896af7a039fe6eece219f892"
    IEEE_API_KEY = "26eer55wvd8r8jy57rhgyjej"
    SEMANTIC_SCHOLAR_API_KEY = "wUzeYkFhlI42jo50TWZ5Blshy1CTxlv6IeqUIoWa"

    parser = argparse.ArgumentParser(description="Healthcare Cybersecurity Literature Collector")
    parser.add_argument("--build-taxonomy", action="store_true", help="Build taxonomy and generate wordcloud")
    parser.add_argument("--suggest-terms", action="store_true", help="Suggest and optionally add new search terms from content")
    parser.add_argument("--build-tree", action="store_true", help="Recursively explore article keywords via DFS")
    parser.add_argument("--tree-api", choices=["semantic_scholar", "pubmed", "arxiv"], default="semantic_scholar", help="Which API to use for tree building")
    parser.add_argument("--fetch-articles", action="store_true", help="Fetch new articles from all APIs and update raw_articles.csv")
    parser.add_argument("--depth", type=int, default=2, help="Depth of recursive keyword exploration")
    parser.add_argument("--query-articles", action="store_true", help="Query raw_articles.csv using filters")
    parser.add_argument("--term", type=str, help="Term to search in title or abstract")
    parser.add_argument("--institution", type=str, help="Institution name to filter by")
    parser.add_argument("--year", type=str, help="Year of publication to filter by")
    parser.add_argument("--keyword", type=str, help="Keyword to filter by")
    parser.add_argument("--limit", type=int, default=10, help="Max number of results to return")

    args = parser.parse_args()
    main(args)