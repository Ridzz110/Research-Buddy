from langgraph.graph import StateGraph, END
from graph.state import ResearchState
from graph.nodes import search_agent, writer_agent, critic_agent, planner_agent, reflect_agent

def should_continue(state: ResearchState) -> str:
    """If reflect produced new queries, loop back. Otherwise end."""
    if state.get("search_queries"):
        return "search"
    return END

def build_graph():

    graph = StateGraph(ResearchState)

    graph.add_node("planner" , planner_agent)
    graph.add_node("search", search_agent)
    graph.add_node("critic", critic_agent)
    graph.add_node("writer", writer_agent)
    graph.add_node("reflect", reflect_agent)

#flow of the program
    graph.set_entry_point("planner")
    graph.add_edge("planner", "search")
    graph.add_edge("search", "critic")
    graph.add_edge("critic", "writer")
    graph.add_edge("writer", "reflect")
    graph.add_conditional_edges("reflect",
                                should_continue,
                                {
                                    "search":"search",
                                    END:END
                                })

    return graph.compile()

research_graph = build_graph()

