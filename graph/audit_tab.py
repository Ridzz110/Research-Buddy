"""
audit_tab.py — Gradio audit panel for Research Buddy.

Renders the critic's full audit trail as a second tab in the app.
Call build_audit_tab() inside your gr.Blocks() context.
Call update_audit(state) in your research() function to populate it.
"""

import gradio as gr
from graph.state import CriticReport


def _render_kept_sources(report: CriticReport) -> str:
    if not report or not report.kept:
        return "No sources evaluated yet."
    lines = []
    for i, s in enumerate(report.kept, 1):
        if s.composite_score >= 0.7:
            badge = "🟢"
        elif s.composite_score >= 0.4:
            badge = "🟡"
        else:
            badge = "🔴"
        lines.append(
            f"{badge} **Source {i}** — Score: `{s.composite_score:.2f}` "
            f"(relevance `{s.relevance_score:.2f}` · quality `{s.quality_score:.2f}`)\n"
            f"- Domain: `{s.domain or 'unknown'}`\n"
            f"- Reason kept: {s.reason}\n"
            f"- URL: {s.url or 'N/A'}\n"
        )
    return "\n---\n".join(lines)


def _render_dropped_sources(report: CriticReport) -> str:
    if not report or not report.dropped:
        return "No sources were dropped."
    lines = []
    for i, s in enumerate(report.dropped, 1):
        lines.append(
            f"❌ **Dropped {i}**\n"
            f"- Reason: {s.reason}\n"
            f"- URL: {s.url or 'N/A'}\n"
            f"- Preview: {s.content[:200].strip()}..."
        )
    return "\n---\n".join(lines)


def _render_contradictions(contradictions: list[dict]) -> str:
    if not contradictions:
        return "✅ No contradictions detected between sources."
    lines = []
    for i, c in enumerate(contradictions, 1):
        lines.append(
            f"⚠️ **Conflict {i}: {c['topic']}**\n\n"
            f"**Source A** `{c['source_a']}`\n> {c['claim_a']}\n\n"
            f"**Source B** `{c['source_b']}`\n> {c['claim_b']}"
        )
    return "\n\n---\n\n".join(lines)


def _render_citations(citations: list[str]) -> str:
    if not citations:
        return "No citations extracted yet."
    return "\n".join(f"{i}. {url}" for i, url in enumerate(citations, 1))


def _render_summary(report: CriticReport) -> str:
    if not report:
        return "Run a query to see the audit summary."
    return (
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Sources evaluated | {report.total_input} |\n"
        f"| Sources kept | {report.total_kept} |\n"
        f"| Sources dropped | {report.total_dropped} |\n"
        f"| Contradictions flagged | {len(report.contradictions)} |"
    )


def update_audit(state: dict):
    """
    Call this at the end of your research() generator with the final state.
    Returns values in the order build_audit_tab() defined its outputs:
    (summary, kept, dropped, contradictions, citations)
    """
    report: CriticReport = state.get("critic_report")
    contradictions: list = state.get("contradictions", [])
    citations: list = state.get("citations", [])

    return (
        _render_summary(report),
        _render_kept_sources(report),
        _render_dropped_sources(report),
        _render_contradictions(contradictions),
        _render_citations(citations),
    )


def build_audit_tab():
    """
    Call inside gr.Blocks() inside a gr.Tab("Audit").
    Returns the five output components so app.py can wire them up.
    """
    gr.Markdown("### 🔬 Source Quality Audit")
    gr.Markdown(
        "Full transparency into what the critic agent kept, dropped, "
        "and flagged — run a query first."
    )

    summary_output = gr.Markdown(
        value="Run a query to see the audit summary.",
        label="Summary"
    )

    with gr.Accordion("Kept Sources", open=False):
        kept_output = gr.Markdown(value="Run a query first.")

    with gr.Accordion("Dropped Sources", open=False):
        dropped_output = gr.Markdown(value="Run a query first.")

    with gr.Accordion("Conflicting Evidence", open=True):
        contradictions_output = gr.Markdown(value="Run a query first.")

    with gr.Accordion("Citations Used in Report", open=False):
        citations_output = gr.Markdown(value="Run a query first.")

    return summary_output, kept_output, dropped_output, contradictions_output, citations_output