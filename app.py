import re
import gradio as gr
from dotenv import load_dotenv
load_dotenv()
from graph.graph import research_graph
from graph.audit_tab import build_audit_tab, update_audit


def _linkify_citations(report_text: str, citations: list[str]) -> str:
    """Replace [Source N] with clickable [[N]](url) for Gradio Markdown."""
    for i, url in enumerate(citations, 1):
        report_text = report_text.replace(
            f"[Source {i}]",
            f"[[{i}]]({url})"
        )
    return report_text


def research(query: str):
    if not query.strip():
        yield "Please enter a query.", "", "", "", "", "", "", "", ""
        return

    status = ""
    sub_questions = ""
    report = ""
    sources = ""

    audit_summary = ""
    audit_kept = ""
    audit_dropped = ""
    audit_contradictions = ""
    audit_citations = ""

    status_map = {
        "planner": "Planner is breaking down your query...",
        "search":  "Search agent is scouring the web...",
        "critic":  "Critic agent is validating sources...",
        "writer":  "Writer agent is composing the report...",
        "reflect": "Reflect agent is grading the report...",
    }

    final_state = {}

    for chunk in research_graph.stream({"query": query}):
        node_name = list(chunk.keys())[0]
        node_data = chunk[node_name]
        final_state.update(node_data)
        status = status_map.get(node_name, "")

        if node_name == "planner" and "sub_questions" in node_data:
            sub_questions = "\n".join([
                f"{i+1}. {q}" for i, q in enumerate(node_data["sub_questions"])
            ])

        if node_name == "search" and "search_results" in node_data:
            status = f"🔍 Found {len(node_data['search_results'])} results. Validating..."

        if node_name == "critic" and "validated_results" in node_data:
            sources = "\n".join([
                r.split("Source: ")[-1].split("\n")[0]  # extract URL only, not Exa-Score line
                for r in node_data["validated_results"]
                if "Source: " in r
            ])
            audit_summary, audit_kept, audit_dropped, audit_contradictions, audit_citations = update_audit(node_data)

        if node_name == "writer" and "report" in node_data:
            raw_report = node_data["report"]
            citations = final_state.get("citations", [])
            report = _linkify_citations(raw_report, citations)
            status = "✅ Report ready!"
            audit_summary, audit_kept, audit_dropped, audit_contradictions, audit_citations = update_audit(final_state)

        if node_name == "reflect" and "reflection" in node_data:
            if node_data.get("search_queries"):
                status = "🔄 Report needs improvement, searching again..."
            else:
                status = "✅ Report passed quality check!"

        yield (
            status,
            sub_questions,
            report,
            sources,
            audit_summary,
            audit_kept,
            audit_dropped,
            audit_contradictions,
            audit_citations,
        )


with gr.Blocks(title="Research Buddy") as app:
    gr.Markdown("# 🔍 Research Buddy")
    gr.Markdown("Multi-agent AI research assistant. Powered by LangGraph + Groq + Exa")

    with gr.Column():
        query_input = gr.Textbox(
            placeholder="e.g. What are the latest breakthroughs in quantum computing?",
            label="Research Query",
            scale=4
        )
        submit_btn = gr.Button("Research", variant="primary", scale=1)

    status_box = gr.Textbox(label="Agent Status", interactive=False)

    with gr.Tabs():
        with gr.Tab("📄 Report"):
            with gr.Column():
                sub_q_output = gr.Textbox(label="Sub Questions", lines=5)
                sources_output = gr.Textbox(label="Sources", lines=5)
                report_output = gr.Markdown(
                    label="Report",
                    buttons=['copy']
                )

        with gr.Tab("🔬 Source Audit"):
            audit_summary, audit_kept, audit_dropped, audit_contradictions, audit_citations = build_audit_tab()

    all_outputs = [
        status_box,
        sub_q_output,
        report_output,
        sources_output,
        audit_summary,
        audit_kept,
        audit_dropped,
        audit_contradictions,
        audit_citations,
    ]

    submit_btn.click(fn=research, inputs=query_input, outputs=all_outputs)
    query_input.submit(fn=research, inputs=query_input, outputs=all_outputs)


if __name__ == "__main__":
    app.launch(gr.themes.Glass())