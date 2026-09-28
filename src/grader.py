import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

GRADER_MODEL = "llama-3.1-8b-instant"


class GradeDocument(BaseModel):
    """Binary relevance score for a retrieved chunk"""

    binary_score: str = Field(
        description="Is the document relevant to the question? Answer 'yes' or 'no'."
    )


_llm = ChatGroq(model=GRADER_MODEL, api_key=os.getenv("GROQ_API_KEY"), temperature=0)
_structured_llm = _llm.with_structured_output(GradeDocument)

_GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader assessing relevance of a retrieved medical literature "
            "excerpt to a user's clinical question.\n"
            "Grade 'yes' only if the excerpt contains information that helps answer "
            "the question. Grade 'no' if it is off-topic or only tangentially related. "
            "Be strict - a partial keyword match is not enough.",
        ),
        ("human", "Retrieved excerpt:\n\n{document}\n\nUser question: {question}"),
    ]
)

_grader = _GRADE_PROMPT | _structured_llm


def grade_documents(question: str, documents: list[Document]) -> list[Document]:
    """Filter retrieved chunks down to the ones an LLM judges relevant"""
    relevant = []
    for doc in documents:
        result = _grader.invoke({"document": doc.page_content, "question": question})
        if result.binary_score.strip().lower().startswith("y"):
            relevant.append(doc)
    return relevant
