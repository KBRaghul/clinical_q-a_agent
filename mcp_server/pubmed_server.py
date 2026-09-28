"""
PubMed MCP server.

Exposes a single tool, `search_pubmed`, that live-searches PubMed via NCBI
Entrez and returns title/abstract/journal/year/pmid for each hit. The agent
invokes this over MCP when local ChromaDB retrieval is graded insufficient.

Run standalone for testing:
    python mcp_server/pubmed_server.py
"""

import os

from dotenv import load_dotenv
from Bio import Entrez
from mcp.server.fastmcp import FastMCP

load_dotenv()

Entrez.email = os.getenv("NCBI_EMAIL")
if os.getenv("NCBI_API_KEY"):
    Entrez.api_key = os.getenv("NCBI_API_KEY")

mcp = FastMCP("pubmed")


@mcp.tool()
def search_pubmed(query: str, max_results: int = 5) -> list[dict]:
    """
    Search PubMed live and return the top matching abstracts.

    Args:
        query: A PubMed search query (plain text or MeSH-qualified).
        max_results: Maximum number of abstracts to return.
    """
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    search_record = Entrez.read(handle)
    handle.close()

    pmids = search_record["IdList"]
    if not pmids:
        return []

    handle = Entrez.efetch(db="pubmed", id=pmids, rettype="abstract", retmode="xml")
    data = Entrez.read(handle)
    handle.close()

    results = []
    for article in data.get("PubmedArticle", []):
        try:
            medline = article["MedlineCitation"]
            pmid = str(medline["PMID"])
            article_data = medline["Article"]
            title = str(article_data.get("ArticleTitle", ""))

            abstract_parts = article_data.get("Abstract", {}).get("AbstractText", [])
            abstract = " ".join(str(part) for part in abstract_parts)
            if not abstract:
                continue

            journal = str(article_data.get("Journal", {}).get("Title", ""))
            pub_date = (
                article_data.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            )
            year = str(pub_date.get("Year", pub_date.get("MedlineDate", "")))

            results.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "journal": journal,
                    "year": year,
                }
            )
        except (KeyError, IndexError):
            continue

    return results


if __name__ == "__main__":
    mcp.run(transport="stdio")
