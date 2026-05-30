from graph.state import ResearchState
from graph.nodes.shared_utils import _invoke_with_fallback

# NODE 1 — PLANNER

def planner_agent(state: ResearchState) -> ResearchState:
    prompt = f"""You are a research planner. Break the following query into exactly 3 focused sub-questions that together will fully answer it.

                Query: {state['query']}

                Return ONLY a numbered list like:
                1. question one
                2. question two
                3. question three"""
    content = _invoke_with_fallback(prompt, max_tokens=750)
    lines = content.strip().split("\n")
    sub_questions = [
        line.split(". ", 1)[-1].strip()
        for line in lines if line.strip() and line[0].isdigit()
    ]
    return {"sub_questions": sub_questions}