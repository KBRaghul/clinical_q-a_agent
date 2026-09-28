import json
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from agent import build_agent  # noqa: E402

app = FastAPI(title="Clinical Q&A Agent")


class QuestionRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
async def ask(request: QuestionRequest):
    """
    Streams the agent's progress as newline-delimited JSON events:
    node-by-node status updates, then the final grounded answer.
    """

    async def event_stream():
        agent = build_agent()
        initial_state = {
            "question": request.question,
            "retrieved_chunks": [],
            "graded_chunks": [],
            "used_web_search": False,
            "answer": "",
        }

        async for step in agent.astream(initial_state):
            for node_name, node_state in step.items():
                if node_name == "generate":
                    event = {"type": "answer", "content": node_state["answer"]}
                else:
                    event = {"type": "status", "node": node_name}
                yield json.dumps(event) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
