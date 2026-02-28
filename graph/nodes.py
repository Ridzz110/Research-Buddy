from langchain_groq import ChatGroq
from tavily import TavilyClient
from graph.state import ResearchState
import os

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    api_key = os.getenv("GROQ_API_KEY")
)

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

##Node one is a planner agent.

def planner_agent(state: ResearchState) -> ResearchState:
    prompt = f"""You are a research planner. Break the following query into exactly 3 focused sub-questions that together will fully answer it.
    
                Query: {state['query']}

                Return ONLY a numbered list like:
                1. question one
                2. question two
                3. question three"""
    response = llm.invoke(prompt)
    lines = response.content.strip().split("\n")
    sub_questions = [
        line.split(". ", 1)[-1].strip()
        for line in lines if line.strip() and line[0].isdigit()
    ]
    return {"sub_questions" : sub_questions}

##node 2 is a search agent.
def search_agent(state: ResearchState) -> ResearchState:
    queries = state.get("search_queries") or state.get("sub_questions", [])
    search_results = state.get("search_results", [])
    for question in queries:
        results = tavily.search(query= question, max_results=3)
        for r in results["results"]:
            formatted = f"**{r['title']}**\n{r['content']}\nSource: {r['url']}"
            search_results.append(formatted)
    
    return {"search_results": search_results}

##node 3 is a search validating agent.
def critic_agent(state: ResearchState) -> ResearchState:
    results_text = "\n\n---\n\n".join(state["search_results"])
    prompt = f"""You are a research critic. Review the following search results for the query: "{state['query']}"

        Search Results:
        {results_text}

        Your job:
        - Remove results that are irrelevant or low quality
        - Flag any contradictions between sources
        - Return only the results worth using in a final report

        Return the filtered results in the same format, separated by ---"""
    response = llm.invoke(prompt)
    validated = [
        r.strip()
        for r in response.content.strip().split("---")
        if r.strip()
    ]
    
    return {"validated_results": validated}

##node 4 is a writer agent.
def writer_agent(state: ResearchState) -> ResearchState:
    results_text = "\n\n---\n\n".join(state["validated_results"])
    prompt = f"""You are a research writer. Using the validated research below, write a comprehensive, well-structured report answering this query: "{state['query']}"

    Research:
    {results_text}

    Format the report in markdown with:
    - A clear title
    - An executive summary (2-3 sentences)
    - Sections with headers for each major finding
    - A sources section at the end listing the URLs used

    Write clearly and concisely."""

    response = llm.invoke(prompt)

    return {"report": response.content}

#node 5 is a feedback loop that grades the final outcome and based it decides if it needs to go back
def reflect_agent(state: ResearchState) -> ResearchState:
    print("DEBUG REPORT IN REFLECT:", state.get("report", "EMPTY"))
    prompt = f"""You are a research quality reviewer. Evaluate the following research report for the query: "{state['query']}"

        Report:
        {state['report']}

        Grade the report on:
        1. Comprehensiveness — does it fully answer the query?
        2. Source quality — are the findings well supported?

        Respond in this exact format:
        GRADE: PASS or FAIL
        REASON: one sentence explanation
        NEW_QUERIES: (only if FAIL) 2 new search queries separated by | to fill the gaps"""
    
    response = llm.invoke(prompt)
    content = response.content.strip()
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
        grade = 'PASS'

    return {
        "reflection": reflection,
        "search_queries": new_queries if grade == "FAIL" else [],
        "iteration": iteration,
        "report": state["report"] if grade == "PASS" else ""
    }