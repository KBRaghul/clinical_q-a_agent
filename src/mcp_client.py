import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_SERVER_SCRIPT = str(Path(__file__).resolve().parent.parent / "mcp_server" / "pubmed_server.py")


async def search_pubmed_live(query: str, max_results: int = 5) -> list[dict]:
    """
    Spawn the PubMed MCP server over stdio, call its search_pubmed tool, and
    return the parsed results. Used as the retrieval fallback when local
    ChromaDB results are graded insufficient.
    """
    params = StdioServerParameters(command=sys.executable, args=[_SERVER_SCRIPT])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "search_pubmed", {"query": query, "max_results": max_results}
            )

    text = result.content[0].text if result.content else "[]"
    return json.loads(text)
