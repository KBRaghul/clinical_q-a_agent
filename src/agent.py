import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from retriever import retrieve
from grader import grade_documents
from mcp_client import search_pubmed_live

load_dotenv()

GENERATION_MODEL = "llama-3.1-8b-instant"
MIN_RELEVANT_CHUNKS = 2  # below this, fall back to live PubMed search via MCP

llm = ChatGroq(model=GENERATION_MODEL, api_key=os.getenv("GROQ_API_KEY"), temperature=0.2)


class AgentState(TypedDict):
    question: str
    retrieved_chunks: list[Document]
    graded_chunks: list[Document]
    used_web_search: bool
    answer: str


# ─── Node: RETRIEVE ────────────────────────────────────────
def retrieve_node(state: AgentState) -> AgentState:
    chunks = retrieve(state["question"], k=4)
    return {**state, "retrieved_chunks": chunks}


# ─── Node: GRADE ───────────────────────────────────────────
def grade_node(state: AgentState) -> AgentState:
    relevant = grade_documents(state["question"], state["retrieved_chunks"])
    return {**state, "graded_chunks": relevant}


def route_after_grading(state: AgentState) -> str:
    if len(state["graded_chunks"]) >= MIN_RELEVANT_CHUNKS:
        return "generate"
    return "web_search_fallback"


# ─── Node: WEB SEARCH FALLBACK (PubMed MCP) ────────────────
async def web_search_fallback_node(state: AgentState) -> AgentState:
    results = await search_pubmed_live(state["question"], max_results=5)

    fetched_docs = [
        Document(
            page_content=f"Title: {r['title']}\n\nAbstract: {r['abstract']}",
            metadata={
                "pmid": r["pmid"],
                "title": r["title"],
                "journal": r["journal"],
                "year": r["year"],
                "source": "pubmed_mcp_live",
            },
        )
        for r in results
    ]

    combined = state["graded_chunks"] + fetched_docs
    return {**state, "graded_chunks": combined, "used_web_search": True}


# ─── Node: GENERATE ────────────────────────────────────────
def generate_node(state: AgentState) -> AgentState:
    context_blocks = []
    for doc in state["graded_chunks"]:
        meta = doc.metadata
        citation = f"PMID {meta.get('pmid', 'unknown')} ({meta.get('journal', '')}, {meta.get('year', '')})"
        context_blocks.append(f"[{citation}]\n{doc.page_content}")
    context = "\n\n".join(context_blocks)

    prompt = f"""You are a clinical literature assistant. Answer the question using
ONLY the information in the provided excerpts. Every claim must be traceable to a
provided excerpt. Cite the PMID for each claim you make, e.g. (PMID 12345678).
If the excerpts don't contain enough information to answer confidently, say so
explicitly instead of guessing.

Excerpts:
{context}

Question: {state['question']}

Answer:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {**state, "answer": response.content.strip()}


# ─── Build the Graph ───────────────────────────────────────
def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("web_search_fallback", web_search_fallback_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        route_after_grading,
        {"generate": "generate", "web_search_fallback": "web_search_fallback"},
    )
    graph.add_edge("web_search_fallback", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


async def ask(question: str) -> AgentState:
    agent = build_agent()
    return await agent.ainvoke(
        {
            "question": question,
            "retrieved_chunks": [],
            "graded_chunks": [],
            "used_web_search": False,
            "answer": "",
        }
    )


if __name__ == "__main__":
    import asyncio

    result = asyncio.run(ask("What are first-line treatments for heart failure with reduced ejection fraction?"))
    print(f"\nUsed web search fallback: {result['used_web_search']}")
    print(f"\nAnswer:\n{result['answer']}")
