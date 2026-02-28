from typing import TypedDict, List

class ResearchState(TypedDict,total=False):
    query: str
    sub_questions: List[str]
    search_results: List[str]
    validated_results: List[str]
    report: str
    reflection: str
    search_queries: List[str]
    iteration: int