# IR-System
IR System to Crawl, Search and recommend based on URL, local dataset path and uploading file

This project implements the below IR workflow:
- Crawling of the URL
- Preprocessing + inverted index
- Vectorizing the documents
- Search function using PageRank and cosine similarity and comparing them.
- Recommendation types
- Evaluation metrices
- Inference and discussion panel

## Project Structure
- `app.py` → Streamlit application
- `requirements.txt` → dependencies
- `data/*.csv` → bundled sample dataset
- `crawled docs/*.txt` → text documents containing contents of crawled URLs
- `Report.md` → implementation report + inferences

## Setup
1. Create virtual environment (recommended)
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run app:
   - `streamlit run app.py`

## Notes
- You can use bundled dataset or upload your own `.csv` files from the UI.
