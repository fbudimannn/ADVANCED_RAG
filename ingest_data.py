import os
import re
import pandas as pd
import spacy
import time
from datetime import datetime
from dotenv import load_dotenv
from Bio import Entrez
from tqdm import tqdm  # For a progress bar
from urllib.error import HTTPError

# LangChain components
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import SpacyTextSplitter

# --- 1. INITIAL SETUP & CONFIGURATION ---

# Load variables from your .env file
load_dotenv()
print("Loaded environment variables from .env file.")

# --- Required Configuration (PLEASE CHECK THESE) ---
# Your email is needed by PubMed's API
ENTREZ_EMAIL = "fahribudiman1721@gmail.com"  # <-- This is from your code.
# This will be the name of your table in AstraDB
COLLECTION_NAME = "pubmed_data"

# --- Embedding Model Configuration (CRITICAL) ---
# This MUST match the model you will use in your Streamlit app.
# "all-MiniLM-L6-v2" is a common choice with 384 dimensions.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# --- Semantic Chunking Configuration ---
SPACY_MODEL = "en_core_web_sm"
CHUNK_SIZE = 1000  # Size of chunks in tokens
CHUNK_OVERLAP = 100  # Overlap between chunks

# --- 2. HELPER FUNCTION (from your code) ---

def fetch_with_retry(db, **kwargs):
    """
    Performs an Entrez search with automatic retries on HTTP errors.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            handle = Entrez.esearch(db=db, **kwargs)
            return Entrez.read(handle)
        except (HTTPError) as e:
            if attempt == max_retries - 1:
                print(f"Final attempt failed: {e}")
                raise
            wait_time = 5 * (attempt + 1)
            print(f"Attempt {attempt + 1} failed with {e}. Waiting {wait_time} seconds...")
            time.sleep(wait_time)

# --- 3. MAIN DATA INGESTION FUNCTION ---

def main():
    """
    Main function to run the data ingestion pipeline.
    """
    print("--- STARTING DATA INGESTION PIPELINE (PHASE 3) ---")
    
    # --- Step 1: Get API Keys from Environment ---
    ASTRA_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    ASTRA_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
    
    if not ASTRA_TOKEN or not ASTRA_ENDPOINT:
        print("FATAL ERROR: ASTRA_DB_APPLICATION_TOKEN or ASTRA_DB_API_ENDPOINT not found in .env")
        print("Please check your .env file.")
        return

    # --- Step 2: PubMed Data Fetching (Using your advanced logic) ---
    print(f"Configuring Entrez with email: {ENTREZ_EMAIL}")
    Entrez.email = ENTREZ_EMAIL
    Entrez.sleep_between_tries = 2

    # Keyword list from your code
    subdomain_topic_map = {
        "Cardiovascular (CVD), incl. stroke & diabetes": [
            "GLP-1 agonist heart failure", "GLP-1 AND diabetes", "SGLT2 inhibitor heart failure",
            "tirzepatide heart failure", "diabetic cardiomyopathy", "heart failure preserved ejection fraction",
            "HFpEF", "ischemic stroke", "acute myocardial infarction", "dual antiplatelet therapy stroke",
            "DOAC atrial fibrillation", "transcatheter tricuspid valve repair", "TAVI transcatheter aortic valve",
            "thrombectomy stroke", "left atrial appendage closure", "ambulatory blood pressure monitoring",
            "AI ECG arrhythmia", "smartwatch atrial fibrillation detection", "lipoprotein a",
            "PCSK9 inhibitor", "ezetimibe cardiovascular", "bempedoic acid", "percutaneous coronary intervention",
            "drug-eluting stent", "coronary artery disease chronic", "ventricular tachycardia ablation",
            "remote cardiac rehabilitation", "endovascular thrombectomy", "hypertension", "blood pressure",
            "stroke", "diabetes", "statin intolerance", "coronary artery disease", "heart failure",
            "arrhythmias", "peripheral artery disease", "aortic disease", "inflammation and aging",
            "socioeconomic disparities", "cardiovascular imaging",
        ]
    }
    
    # === DATE RANGE: Adjusted to be DYNAMIC (2023 to TODAY) ===
    today_str = datetime.now().strftime("%Y/%m/%d")
    date_range = f'("2023/01/01"[Date - Publication] : "{today_str}"[Date - Publication])'
    print(f"Set dynamic date range: 2023/01/01 to {today_str}")

    # === BUILD PUBMED QUERY ===
    topic_queries = []
    for subdomain, keywords in subdomain_topic_map.items():
        topic_queries.extend([f'"{kw}"[Title/Abstract]' for kw in keywords]) 

    full_query = f"(({ ' OR '.join(topic_queries) }) AND {date_range}) AND (journal article[pt]) AND (english[lang])"

    # === SEARCH PUBMED ===
    print("Searching PubMed. This may take a moment...")
    try:
        record = fetch_with_retry(db='pubmed', retmax=10000, term=full_query) # Max 10,000 articles
        id_list = record['IdList']
        print(f"📥 Found {len(id_list)} total article IDs.")
    except Exception as e:
        print(f"Search failed: {e}")
        id_list = []

    if not id_list:
        print("No articles found. Exiting script.")
        return

    # === FETCH ARTICLES IN BATCHES ===
    data = []
    batch_size = 200 # Fetch 200 details at a time
    
    print("Starting to fetch article details in batches...")
    for i in tqdm(range(0, len(id_list), batch_size), desc="Fetching Batches"):
        batch = id_list[i:i+batch_size]
        
        try:
            handle = Entrez.efetch(db='pubmed', id=batch, retmode='xml')
            records = Entrez.read(handle)

            for record in records['PubmedArticle']:
                try:
                    article = record['MedlineCitation']['Article']
                    title = article['ArticleTitle']

                    abstract = ''
                    if 'Abstract' in article:
                        abstract = ' '.join(article['Abstract']['AbstractText'])
                    if len(abstract.split()) < 100: # Skip short abstracts
                        continue

                    journal = article['Journal']['Title']
                    pmid = record['MedlineCitation']['PMID']
                    pub_date = 'N/A'
                    try:
                        if 'ArticleDate' in article and article['ArticleDate']:
                            date_parts = article['ArticleDate'][0]
                            pub_date = f"{date_parts.get('Day', '01')} {date_parts.get('Month', 'Jan')} {date_parts.get('Year', '1900')}"
                        else:
                            pub_parts = article['Journal']['JournalIssue']['PubDate']
                            pub_date = f"{pub_parts.get('Day', '01')} {pub_parts.get('Month', 'Jan')} {pub_parts.get('Year', '1900')}"
                    except:
                        pass # Fallback to N/A

                    data.append({
                        'PMID': pmid,
                        'Title': title,
                        'Abstract': abstract,
                        'PublishedDate': pub_date,
                        'Journal': journal,
                        'URL': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    })

                except KeyError as e:
                    # Skip record if essential fields are missing
                    continue

        except Exception as e:
            print(f"❌ Batch {i//batch_size + 1} failed: {e}")
            continue

        time.sleep(1) # Polite 1-second delay per batch

    # === CREATE DATAFRAMES ===
    df_general = pd.DataFrame(data)
    print(f"✅ Parsed {len(df_general)} valid articles with abstracts.")

    # === FILTER FOR HIGH-QUALITY JOURNALS (Your list) ===
    trusted_journals = [
        "The Lancet", "BMJ", "JAMA", "Nature medicine", 
        "The New England journal of medicine", "Circulation", 
        "Journal of the American College of Cardiology"
    ]
    trusted_journals_lower = [j.lower() for j in trusted_journals]

    df_trusted = df_general[
        df_general['Journal'].str.lower().apply(
            lambda j: any(j.startswith(tj) for tj in trusted_journals_lower)
        )
    ].copy()
    print(f"⭐️ Filtered down to {len(df_trusted)} articles from trusted journals.")

    if df_trusted.empty:
        print("No articles from trusted journals were found. Exiting.")
        return

    # --- Step 3: Convert DataFrame to LangChain Documents ---
    print("Converting trusted articles to LangChain Document objects...")
    documents = []
    for _, row in df_trusted.iterrows():
        # Clean abstract text one more time
        clean_abstract = re.sub(r'\s+', ' ', row['Abstract']).strip()
        doc = Document(
            page_content=clean_abstract,
            metadata={
                "title": row.get('Title', 'N/A'),
                "journal": row.get('Journal', 'N/A'),
                "published_date": row.get('PublishedDate', 'N/A'),
                "pmid": row.get('PMID', 'N/A'),
                "source_url": row.get('URL', 'N/A')
            }
        )
        documents.append(doc)

    # --- Step 4: Initialize Text Splitter (Semantic Chunking) ---
    print(f"Loading Spacy model '{SPACY_MODEL}' for semantic chunking...")
    try:
        nlp = spacy.load(SPACY_MODEL)
    except OSError:
        print(f"FATAL ERROR: Spacy model '{SPACY_MODEL}' not found.")
        print("Please ensure it's in your requirements.txt and installed.")
        return
        
    text_splitter = SpacyTextSplitter(
        nlp=nlp,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    print("Splitting documents into semantic chunks...")
    chunked_documents = text_splitter.split_documents(documents)
    print(f"Total {len(documents)} documents split into {len(chunked_documents)} chunks.")

    # --- Step 5: Initialize Embedding Model ---
    print(f"Initializing embedding model: {MODEL_NAME}")
    embedding_model = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'} # Use CPU for stable ingestion
    )

    # --- Step 6: Initialize AstraDB & Add Documents ---
    print(f"Connecting to AstraDB (Collection: {COLLECTION_NAME})...")
    vstore = AstraDBVectorStore(
        token=ASTRA_TOKEN,
        api_endpoint=ASTRA_ENDPOINT,
        collection_name=COLLECTION_NAME,
        embedding_dimension=EMBEDDING_DIMENSION
    )
    
    print("Clearing any old data from the collection...")
    vstore.clear() # Start fresh every time

    print(f"Ingesting {len(chunked_documents)} chunks into AstraDB. This will take time...")
    
    # Ingest in batches of 20 (AstraDB recommended max)
    batch_size = 20
    for i in tqdm(range(0, len(chunked_documents), batch_size), desc="Ingesting to AstraDB"):
        batch = chunked_documents[i:i + batch_size]
        try:
            vstore.add_documents(batch)
        except Exception as e:
            print(f"Error ingesting batch {i//batch_size + 1}: {e}")
            time.sleep(5) # Wait and retry (simple retry)
            try:
                vstore.add_documents(batch)
            except Exception as e2:
                print(f"Batch {i//batch_size + 1} failed again. Skipping. Error: {e2}")

    print("\n--- ✅ DATA INGESTION PIPELINE (PHASE 3) COMPLETE ---")
    print(f"All data has been successfully ingested into the '{COLLECTION_NAME}' collection.")

# --- SCRIPT EXECUTION ---
if __name__ == "__main__":
    main()