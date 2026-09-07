import os
import json
import time

from dotenv import load_dotenv
from Bio import Entrez
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Constants
QUERY = '"heart failure"[MeSH] OR "myocardial infarction"[MeSH] OR "arrhythmias, cardiac"[MeSH]'
MAX_RESULTS = 100  # bump to 1000+ once the pipeline is verified
CACHE_PATH = "data/pubmed_cardiology.json"
CHROMA_DIR = ".chroma"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

Entrez.email = os.getenv("NCBI_EMAIL")
if os.getenv("NCBI_API_KEY"):
    Entrez.api_key = os.getenv("NCBI_API_KEY")


def search_pmids(query: str, max_results: int) -> list[str]:
    """Search PubMed and return matching PMIDs"""
    print(f"Searching PubMed: {query}")
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    pmids = record["IdList"]
    print(f"Found {len(pmids)} PMIDs")
    return pmids


def fetch_abstracts(pmids: list[str], batch_size: int = 50) -> list[dict]:
    """Fetch title/abstract/journal/date for each PMID, in batches"""
    records = []

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        print(f"Fetching batch {i // batch_size + 1} ({len(batch)} PMIDs)...")

        handle = Entrez.efetch(db="pubmed", id=batch, rettype="abstract", retmode="xml")
        data = Entrez.read(handle)
        handle.close()

        for article in data["PubmedArticle"]:
            try:
                medline = article["MedlineCitation"]
                pmid = str(medline["PMID"])
                article_data = medline["Article"]
                title = str(article_data.get("ArticleTitle", ""))

                abstract_parts = article_data.get("Abstract", {}).get("AbstractText", [])
                abstract = " ".join(str(part) for part in abstract_parts)

                if not abstract:
                    continue  # skip entries with no abstract text

                journal = str(article_data.get("Journal", {}).get("Title", ""))
                pub_date = (
                    article_data.get("Journal", {})
                    .get("JournalIssue", {})
                    .get("PubDate", {})
                )
                year = str(pub_date.get("Year", pub_date.get("MedlineDate", "")))

                records.append(
                    {
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "journal": journal,
                        "year": year,
                    }
                )
            except (KeyError, IndexError):
                continue  # skip malformed records

        time.sleep(0.4)  # stay under NCBI rate limits

    print(f"Fetched {len(records)} abstracts with usable text")
    return records


def load_cached_or_fetch() -> list[dict]:
    """Reuse a cached fetch if present, otherwise hit PubMed"""
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached abstracts from {CACHE_PATH}")
        with open(CACHE_PATH) as f:
            return json.load(f)

    pmids = search_pmids(QUERY, MAX_RESULTS)
    records = fetch_abstracts(pmids)

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Cached raw abstracts to {CACHE_PATH}")

    return records


def to_documents(records: list[dict]) -> list[Document]:
    """Convert raw records into LangChain Documents with citation metadata"""
    documents = []
    for r in records:
        content = f"Title: {r['title']}\n\nAbstract: {r['abstract']}"
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "pmid": r["pmid"],
                    "title": r["title"],
                    "journal": r["journal"],
                    "year": r["year"],
                },
            )
        )
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Chunk documents - abstracts are short, so use a modest chunk size"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} abstracts into {len(chunks)} chunks")
    return chunks


def store_in_chromadb(chunks: list[Document]):
    """Embed and persist chunks to ChromaDB"""
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Storing vectors in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"Stored {len(chunks)} chunks in {CHROMA_DIR}")
    return vectorstore


def main():
    print("=== Starting PubMed Ingestion (cardiology) ===\n")

    records = load_cached_or_fetch()
    documents = to_documents(records)
    chunks = split_documents(documents)
    store_in_chromadb(chunks)

    print("\n=== Ingestion Complete ===")


if __name__ == "__main__":
    main()
