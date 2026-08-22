
#importing necessary libraries
import os
import streamlit as st
import requests, re
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
from urllib.parse import urljoin, urlparse


#Removing Stopwords
stopwords = ENGLISH_STOP_WORDS

def clean_text(text):
    text = re.sub(r'\W+', ' ', str(text).lower())
    return " ".join([w for w in text.split() if w not in stopwords])


#Removing duplicate documents
def remove_duplicate(docs):
    seen = set()
    unique_docs = []
    for d in docs:
        if d["id"] not in seen:
            unique_docs.append(d)
            seen.add(d["id"])
    return unique_docs


#vectorizing the documents
def vectorizing(docs):
    tf_idf = TfidfVectorizer()
    x = tf_idf.fit_transform([clean_text(d["content"]) for d in docs])
    return x, tf_idf


#Search function using Page Rank

#ranking the docs using pagerank
def pagerank_graph(docs, X, top_k =10, threshold=0.2):
    G = nx.DiGraph()
    for i, d1 in enumerate(docs):
        G.add_node(d1["id"])
        # Compute similarities for doc i against all docs
        sims = cosine_similarity(X[i], X).flatten()
        # Get top-N neighbors (excluding itself)
        top_neighbors = np.argsort(sims)[::-1][1:top_k+1]
        for j in top_neighbors:
            if sims[j] > threshold:
                G.add_edge(d1["id"], docs[j]["id"], weight=sims[j])
    return nx.pagerank(G, weight="weight")

#Function to search the document
def search(query, tf_idf, X, docs, pagerank_scores, top_k=5):
    q_vec = tf_idf.transform([query])
    scores = cosine_similarity(q_vec, X).flatten()

    # PageRank scores vector aligned with docs
    y_scores = np.array([pagerank_scores.get(doc["id"], 0) for doc in docs])

    # If all scores are zero, fallback to keyword search
    if np.all(scores == 0):

        results = []
        for doc in docs:
            content = doc.get("content", "").lower()
            title = str(doc["metadata"].get("title", "")).lower()
            if query.lower() in content or query.lower() in title:
                keyword_score = content.count(query.lower()) + title.count(query.lower())
                results.append({
                    "id": doc["id"],
                    "snippet": doc["content"][:200],
                    "cosine_score": keyword_score,
                    "pagerank_score": pagerank_scores.get(doc["id"], 0),
                    "final_score": (0.7 * keyword_score) + (0.3 * pagerank_scores.get(doc["id"], 0)),
                    "metadata": doc.get("metadata", {})
                })
        return sorted(results, key=lambda r: r["cosine_score"], reverse=True)[:top_k], y_scores

    # Normal TF‑IDF ranking
    ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)
    results = []
    alpha = 0.7
    for i, _ in ranked[:top_k]:
        doc = docs[i]
        cos_score = scores[i]
        pg_rnk_score = pagerank_scores.get(doc["id"], 0)
        final_score =   alpha * cos_score + (1 - alpha) * pg_rnk_score
        results.append({
            "id": doc["id"],
            "snippet": doc["content"][:200],
            "cosine_score": round(cos_score, 3),
            "pagerank_score": round(pg_rnk_score, 3),
            "final_score": round(final_score, 3),
            "metadata": doc.get("metadata", {})
        })
    ranked = sorted(results, key=lambda r: r["final_score"], reverse=True)
    return ranked[:top_k], y_scores


#Recomendation Modules

#Content Based recomendation
def content_based_rec(user_doc, tf_idf, X, docs, top_k = 5):
    u_vec = tf_idf.transform([user_doc])
    cos_sim = cosine_similarity(u_vec, X).flatten()
    ranked = sorted(list(enumerate(cos_sim)), key=lambda x: x[1], reverse=True)
    return [(docs[i]["id"], docs[i]["content"][:200], round(cos_sim[i], 3)) for i, _ in ranked[:top_k]]

#Collaborative recomendation
def collaborative_rec(user_id, ratings_matrix, top_k=5):
    cos_sim = cosine_similarity(ratings_matrix)
    similar_users = np.argsort(cos_sim[user_id])[::-1]
    recommendations = {}
    for u in similar_users:
        items = np.where(ratings_matrix[u] > 0)[0]
        for item in items:
            if ratings_matrix[user_id, item] == 0:
                recommendations[item] = cos_sim[user_id, u]
    ranked = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

#Hybrid recommendation
def hybrid_rec(content_scores, collab_scores, alpha=0.5, top_k=5):
    hybrid = {}
    for item, score in content_scores.items():
        hybrid[item] = alpha * score + (1-alpha) * collab_scores.get(item, 0)
    ranked = sorted(hybrid.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

#Function to create a document dictonary with content  and metadata
def make_doc(doc_id, content, metadata=None):
    return {
        "id": doc_id,
        "content": str(content),
        "metadata": metadata if metadata else {}
    }


#CRAWLING

#normalizing URL
def normalize_url(url):
    parsed = urlparse(url)
    normalized = parsed.scheme + "://" + parsed.netloc + parsed.path
    return normalized.rstrip("/")  


#checking for valid url
def valid_url(url):
    parsed = urlparse(url)
    return parsed.scheme in ["http", "https"]

#crawling Function
def crawl(seed_urls, max_depth=1, max_pages = 50, saved_fol =  "crawled_docs"):
    visited, documents = set(), []
    frontier = deque([(normalize_url(u.strip()), 0) for u in seed_urls if u.strip()])
    page_number = 1

    os.makedirs(saved_fol, exist_ok= True)

    while frontier and len(visited) < max_pages:
        current_url, depth = frontier.popleft()
        current_url = normalize_url(current_url)

        if current_url in visited:
            continue
        if depth > max_depth:
            continue
        if not valid_url(current_url):
            continue

        try:
            resp = requests.get(current_url, timeout=5,
                                headers={"User-Agent": "IR-Assignment-Bot"})
            soup = BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException:
            continue

        text = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
        if len(text) < 100:  
            continue

        visited.add(current_url)

        #Save the page
        file_name = f"Document_{page_number}.txt"
        file_path = os.path.join(saved_fol, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)


        metadata = {
            "title": soup.title.get_text(strip=True) if soup.title else current_url,
            "url": current_url,
            "depth": depth,
            "length": len(text),
            "doc_id": f"Document_{page_number}"
        }
        documents.append(make_doc(metadata["doc_id"], text, metadata))
        page_number += 1

        if depth < max_depth:
            for a in soup.find_all("a", href=True):
                absolute = urljoin(current_url, a["href"]).split("#")[0]
                absolute = normalize_url(absolute)
                if absolute not in visited:
                    frontier.append((absolute, depth + 1))

    return remove_duplicate(documents)

#upload dataset
def upload_dataset(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = df.drop_duplicates()
    text_column = df.columns[0]
    file_name = uploaded_file.name
    docs = []
    for i, row in df.iterrows():
        metadata = {"filename": file_name}
        for col in df.columns:
            if col != text_column:
                metadata[col] = row[col]
        docs.append(make_doc(i, row[text_column], metadata))
    return remove_duplicate(docs)


#Load Dataset
def load_dataset(path, text_column=None):
    df = pd.read_csv(path).drop_duplicates()

    # If no text_column is provided, try to auto-detect
    if text_column is None:
        # Common text-like column names
        candidates = ["message", "text", "description", "content", "overview", "review", "body"]
        for col in df.columns:
            if col.lower() in candidates:
                text_column = col
                break
        # Fallback: use the longest string column
        if text_column is None:
            string_cols = [col for col in df.columns if df[col].dtype == "object"]
            if string_cols:
                # Pick the column with the longest average string length
                text_column = max(string_cols, key=lambda c: df[c].astype(str).str.len().mean())
            else:
                # Last fallback: just use the first column
                text_column = df.columns[0]

    docs = []
    for i, row in df.iterrows():
        metadata = {"source": path}
        for col in df.columns:
            metadata[col] = row[col]
        doc_id = f"{os.path.basename(path)}_{i}"
        docs.append(make_doc(doc_id, row[text_column], metadata))

    return docs





#Evaluation Metrics

#Preccions@K
def precision_at_k(y_true, y_scores, k):
    top_k = np.argsort(y_scores)[::-1][:k]
    return np.sum(y_true[top_k]) / k

#Recall@K
def recall_at_k(y_true, y_scores, k):
    top_k = np.argsort(y_scores)[::-1][:k]
    return np.sum(y_true[top_k]) / np.sum(y_true)

#MAP
def mean_average_precision(y_true, y_scores):
    sorted_idx = np.argsort(y_scores)[::-1]
    relevant = np.where(y_true[sorted_idx] == 1)[0]
    if len(relevant) == 0:
        return 0
    precisions = [(i+1)/(idx+1) for i, idx in enumerate(relevant)]
    return np.mean(precisions)

#MRR
def mean_reciprocal_rank(y_true, y_scores):
    sorted_idx = np.argsort(y_scores)[::-1]
    for i, idx in enumerate(sorted_idx):
        if y_true[idx] == 1:
            return 1.0 / (i+1)
    return 0

#NDCG
def ndcg(y_true, y_scores, k):
    sorted_idx = np.argsort(y_scores)[::-1][:k]
    dcg = np.sum([(y_true[idx]) / np.log2(i+2) for i, idx in enumerate(sorted_idx)])
    ideal_idx = np.argsort(y_true)[::-1][:k]
    idcg = np.sum([(y_true[idx]) / np.log2(i+2) for i, idx in enumerate(ideal_idx)])
    return dcg / idcg if idcg > 0 else 0

#Function for Evaluation Metrics
def eval_ir(y_true, y_scores,  k= 2):
    metrics = {
        "Precision": np.mean(y_true[y_scores > 0.5]) if np.sum(y_scores > 0.5) > 0 else 0,
        "Recall": np.sum(y_true[y_scores > 0.5]) / np.sum(y_true) if np.sum(y_true) > 0 else 0,
        "Precision@K": precision_at_k(y_true, y_scores, k),
        "Recall@K": recall_at_k(y_true, y_scores, k),
        "MAP": mean_average_precision(y_true, y_scores),
        "MRR": mean_reciprocal_rank(y_true, y_scores),
        "NDCG@K": ndcg(y_true, y_scores, k)
    }
    metrics["F1-score"] = (
        2 * (metrics["Precision"] * metrics["Recall"]) / (metrics["Precision"] + metrics["Recall"])
        if metrics["Precision"] + metrics["Recall"] > 0 else 0
    )

    st.write("Evaluation Metrics:")
    st.table(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))
    
    #Comparative visualization
    fig2, ax2 = plt.subplots()
    ax2.bar(metrics.keys(), metrics.values())
    ax2.set_ylabel("Score")
    ax2.set_title("Comparative IR Metrics")
    plt.xticks(rotation=45)
    st.pyplot(fig2)
    return metrics


#Function to get y_true value
def get_y_true(docs, query):

    query = str(query).lower()
    y_true = []

    for doc in docs:
        meta = doc.get("metadata", {})
        content = str(doc.get("content", "")).lower()
        relevant = False

        # Check content
        if query in content:
            relevant = True

        # Check all metadata fields
        for key, value in meta.items():
            if isinstance(value, str) and query in value.lower():
                relevant = True
                break
            # Numeric metadata (depth, length)
            if key == "depth":
                try:
                    if int(value) <= int(query):
                        relevant = True
                        break
                except ValueError:
                    pass
            if key == "length":
                try:
                    if int(value) >= int(query):
                        relevant = True
                        break
                except ValueError:
                    pass

        y_true.append(1 if relevant else 0)

    return np.array(y_true)



#Streamlit UI
st.title("End-to-End Information Retrieval System")

data_choice = st.radio("Choose datasource:", ["Web Crawling", "Local Kaggle Dataset (Path)", "Upload Dataset"])

if "docs" not in st.session_state:
    st.session_state.docs = []
if "X" not in st.session_state:
    st.session_state.X = None
if "tfidf" not in st.session_state:
    st.session_state.tfidf = None
if "queries_ran" not in st.session_state:
    st.session_state.queries_ran = 0



#code to run if Web Crawling is choosed
if data_choice == "Web Crawling":
    seed_urls = st.text_input("Enter seed URLs (comma separated)")
    depth = st.slider("Crawling Depth", 1, 3, 1)
    max_pages =  st.slider("Max Page", 5, 50, 20)
    if st.button("Crawl"):
        docs = crawl(seed_urls.split(","), max_depth=depth, max_pages=max_pages)
        X, tfidf = vectorizing(st.session_state.docs)
        st.session_state.docs = docs
        st.session_state.X, st.session_state.tfidf = vectorizing(docs)
        st.session_state.queries_ran += 1
        st.success(f"Crawled {len(docs)} unique documents")


#Code to run if Dataset Path is choosed
elif data_choice == "Local Kaggle Dataset (Path)":
    dataset_path = st.text_area("Enter dataset path(one per line)")
    if st.button("Load Datasets"):
        docs = []
        for path in dataset_path.splitlines():
            if path.strip():
                try:
                    docs.extend(load_dataset(path.strip(), text_column=None))
                except Exception as e:
                    st.error(f"Failed to load {path}: {e}")
        if docs:
            X, tfidf = vectorizing(docs)
            st.session_state.docs = docs
            st.session_state.X, st.session_state.tfidf = vectorizing(docs)
            st.session_state.queries_ran += 1

            st.success(f"Loaded {len(docs)} unique documents from multiple datasets")
        else:
            st.warning("No valid documents found. Please check your dataset paths or column name.")


#Code to run if upload is choosed
elif data_choice == "Upload Dataset":
    uploaded_files = st.file_uploader("Upload one or more CSV file", type="csv", accept_multiple_files=True)
    if uploaded_files:
        docs = []
        for uploaded_file in uploaded_files:
            try:
                docs.extend(upload_dataset(uploaded_file))
            except Exception as e:
                st.error(f"Failed to process {uploaded_file.name}: {e}")
        if docs:
            X, tfidf = vectorizing(docs)
            st.session_state.docs = docs
            st.session_state.X, st.session_state.tfidf = vectorizing(docs)
            st.session_state.queries_ran += 1

            st.success(f"Prepared {len(docs)} unique documents from uploaded files")
        else:
            st.warning("No valid documents found. Please check your uploaded files.")

#Search
query = st.text_input("Search Query")
if st.button("Search") and st.session_state.docs:
    pagerank_scores = pagerank_graph(st.session_state.docs, st.session_state.X)
    pagerank_scores = {k: v * len(st.session_state.docs) for k, v in pagerank_scores.items()}
    results, y_scores = search(query, st.session_state.tfidf, st.session_state.X, st.session_state.docs, pagerank_scores)

    y_true = get_y_true(st.session_state.docs, query)
    # Call evaluation function
    metrics = eval_ir(y_true, y_scores, k=2)


    st.write("Top Search Results (Cosine vs PageRank):")
    for r in results:
      st.write(f"ID: {r['id']}")
      st.write(f"Snippet: {r['snippet']}")
      st.write(f"Cosine Similarity: {r['cosine_score']}, PageRank: {r['pagerank_score']}")
      if r["metadata"]:
        st.write("Metadata:", r["metadata"])
      st.write("---")

     # Visualization of ranking
    st.write("Ranking Visualization:")
    labels = [r["id"] for r in results]
    cosine_scores = [r["cosine_score"] for r in results]
    pagerank_scores_list = [r["pagerank_score"] for r in results]
    fig, ax = plt.subplots()
    x = range(len(labels))
    ax.bar(x, cosine_scores, width=0.4, label="Cosine Similarity", align="center")
    ax.bar([i+0.4 for i in x], pagerank_scores_list, width=0.4, label="PageRank", align="center")
    ax.set_xticks([i+0.2 for i in x])
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.legend()
    st.pyplot(fig)

# Recommendation Section
rec_doc = st.text_area("Enter text for recommendation")
rec_type = st.radio("Choose Recommendation Type:", ["Content-Based", "Collaborative", "Hybrid"])

if st.button("Recommend") and st.session_state.docs:
    tfidf = st.session_state.tfidf
    u_vec=tfidf.transform([rec_doc])
    y_scores = cosine_similarity(u_vec, st.session_state.X).flatten()
    if rec_type == "Content-Based":
        recs = content_based_rec(rec_doc, st.session_state.tfidf, st.session_state.X, st.session_state.docs, top_k=2)
        y_true = get_y_true(st.session_state.docs, rec_doc)

        pagerank_scores = pagerank_graph(st.session_state.docs, st.session_state.X)
        pagerank_scores = {k: v * len(st.session_state.docs) for k, v in pagerank_scores.items()}

        st.write("Top Content-Based Recommendations:")
        for doc_id, snippet, score in recs:
            doc = next(d for d in st.session_state.docs if d["id"] == doc_id)
            st.write(f"ID: {doc_id}")
            st.write(f"Snippet: {snippet}")
            st.write(f"Similarity Score: {score}")
            st.write(f"Cosine Similarity: {score}, PageRank: {round(pagerank_scores.get(doc_id,0),3)}")
            if doc["metadata"]:
              st.write("Metadata:", doc["metadata"])
        st.write("---")

    elif rec_type == "Collaborative":
        ratings_matrix = np.random.randint(0, 6, size=(10, len(st.session_state.docs)))
        user_id = 0
        collab_recs = collaborative_rec(user_id, ratings_matrix, top_k=5)

        pagerank_scores = pagerank_graph(st.session_state.docs, st.session_state.X)
        pagerank_scores = {k: v * len(st.session_state.docs) for k, v in pagerank_scores.items()}

        st.write("Top Collaborative Recommendations:")
        for item, score in collab_recs:
          doc = st.session_state.docs[item]
          st.write(f"ID: {doc['id']}")
          st.write(f"Snippet: {doc['content'][:200]}")
          st.write(f"Hybrid Score: {round(score,3)}, PageRank: {round(pagerank_scores.get(doc['id'],0),3)}")
          if doc["metadata"]:
              st.write("Metadata:", doc["metadata"])
          st.write("---")

    elif rec_type == "Hybrid":
        u_vec = tfidf.transform([rec_doc])
        content_scores = cosine_similarity(u_vec, st.session_state.X).flatten()
        content_dict = {i: content_scores[i] for i in range(len(st.session_state.docs))}
        ratings_matrix = np.random.randint(0, 6, size=(10, len(st.session_state.docs)))
        user_id = 0
        collab_recs = collaborative_rec(user_id, ratings_matrix, top_k=len(st.session_state.docs))
        collab_dict = {item: score for item, score in collab_recs}
        hybrid_recs = hybrid_rec(content_dict, collab_dict, alpha=0.5, top_k=5)
        u_vec=tfidf.transform([rec_doc])
        y_scores = cosine_similarity(u_vec, st.session_state.X).flatten()

        pagerank_scores = pagerank_graph(st.session_state.docs, st.session_state.X)
        pagerank_scores = {k: v * len(st.session_state.docs) for k, v in pagerank_scores.items()}

        st.write("Top Hybrid Recommendations:")
        for item, score in hybrid_recs:
          doc = st.session_state.docs[item]
          st.write(f"ID: {doc['id']}")
          st.write(f"Snippet: {doc['content'][:200]}")
          st.write(f"Hybrid Score: {round(score,3)}, PageRank: {round(pagerank_scores.get(doc['id'],0),3)}")
          if doc["metadata"]:
              st.write("Metadata:", doc["metadata"])
          st.write("---")
    
    y_true = get_y_true(st.session_state.docs, rec_doc) 
    # Call evaluation function
    metrics = eval_ir(y_true, y_scores, k=5)
    st.write(f"Evaluation Metrics ({rec_type} Recommendation):")
    st.table(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))

# Sidebar Dashboard
st.sidebar.title("Dashboard")
st.sidebar.metric("Documents Indexed", len(st.session_state.docs))
st.sidebar.metric("Vector Size", st.session_state.X.shape[1] if st.session_state.X is not None else 0)
st.sidebar.metric("Queries Run", st.session_state.queries_ran)

# Index Management
if st.sidebar.button("Clear Index"):
    docs, X, tfidf = [], None, None
    st.sidebar.success("Index cleared")
