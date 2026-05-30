from graph.state import ResearchState
from graph.nodes.shared_utils import _invoke_with_fallback

# NODE 5 — REFLECT

def reflect_agent(state: ResearchState) -> ResearchState:
    print("DEBUG REPORT IN REFLECT:", state.get("report", "EMPTY"))
    prompt = f"""You are a research quality reviewer. Evaluate the following research report for the query: "{state['query']}"

        Report:
        {state['report']}

        Grade the report on:
        1. Comprehensiveness - does it fully answer the query?
        2. Source quality - are the findings well supported?

        Respond in this exact format:
        GRADE: PASS or FAIL
        REASON: one sentence explanation
        NEW_QUERIES: (only if FAIL) 2 new search queries separated by | to fill the gaps"""

    content = _invoke_with_fallback(prompt, max_tokens=1024).strip()
    print("DEBUG CONTENT REFLECT:", content)

    grade = "PASS"
    new_queries = []
    reflection = content

    for line in content.split("\n"):
        if line.startswith("GRADE:"):
            grade = line.replace("GRADE:", "").strip()
        if line.startswith("NEW_QUERIES:"):
            raw = line.replace("NEW_QUERIES:", "").strip()
            new_queries = [q.strip() for q in raw.split("|") if q.strip()]

    iteration = state.get("iteration", 0) + 1
    if iteration >= 2:
        grade = "PASS"

    return {
        "reflection": reflection,
        "search_queries": new_queries if grade == "FAIL" else [],
        "iteration": iteration,
        "report": state["report"] if grade == "PASS" else "",
    }