# 🧠 Healthcare Cybersecurity Literature Scraper (`scrappy.py`)

@Sebastian-Frazier

A command-line tool for collecting, organizing, and exploring academic research on **healthcare cybersecurity**. This tool scrapes data from multiple sources, builds topic taxonomies, and performs recursive topic expansion using keyword-based DFS.

---

## 🚀 Features

- Collects academic articles from:
  - IEEE Xplore
  - Semantic Scholar
  - PubMed
  - arXiv
  - Google Scholar (via SerpAPI)
- Merges results into a persistent `raw_articles.csv` (never deletes existing data)
- Prompts for manual and automatic keyword additions
- Builds topic taxonomies using BERTopic and generates word clouds
- Explores semantic keyword trees via DFS up to a user-defined depth
- Command-line interface with flexible flags and options

## 📦 Installation

1. **Create and Activate Virtual Environment**

```bash
python3 -m venv venv 
source venv/bin/activate
```

2. **Install Required Packages**

```bash
pip install bertopic sentence-transformers scikit-learn wordcloud matplotlib tqdm beautifulsoup4 requests pandas nltk lxml numpy==1.23.5
```

## 🔧 CLI Flags

--build-taxonomy --> Build topics using BERTopic and generate a word cloud

--suggest-terms --> Suggest and optionally add new keywords from recent articles

--build-tree --> Recursively explore new keyword spaces using DFS

--tree-api --> API for DFS tree (semantic_scholar, pubmed, or arxiv)

--depth --> Depth of recursive keyword discovery (with --build-tree)

## ⚙️ Usage

### Collect Articles from All APIs

```bash
python3 scrapp.py
```

### Build a Topic Taxonomy and Word Cloud

```bash
python3 scrapp.py --build-taxonomy
```

### Suggest and Add New Search Terms

```bash
python scrappy.py --suggest-terms
```

### Perform Recursive Topic Discovery via DFS

```bash
python scrappy.py --build-tree --tree-api semantic_scholar --depth 3
```
