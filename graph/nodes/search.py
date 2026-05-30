from graph.state import ResearchState
from exa_py import Exa
import os


exa = Exa(api_key=os.environ.get("EXA_API_KEY"))


# NODE 2 — SEARCH


def search_agent(state: ResearchState) -> ResearchState:
    queries = state.get("search_queries") or state.get("sub_questions", [])
    search_results = state.get("search_results", [])

    for question in queries:
        results = exa.search(
            question,
            type="auto",
            num_results=2,
            contents={"highlights": True},
            exclude_domains=[
                "facebook.com", "instagram.com", "twitter.com",
                "x.com", "pinterest.com", "reddit.com", "tiktok.com",
            ]
        )
        for r in results.results:
            highlights = " ".join(r.highlights) if r.highlights else ""
            formatted = (
                f"**{r.title}**\n{highlights}\n"
                f"Source: {r.url}\n"
                f"Exa-Score: {r.score}"
            )
            search_results.append(formatted)

    return {"search_results": search_results}