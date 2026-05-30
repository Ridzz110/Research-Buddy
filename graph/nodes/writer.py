from graph.nodes.shared_utils import _invoke_with_fallback, _extract_urls
import re
from graph.state import ResearchState

# NODE 4 — WRITER


def _format_contradictions_for_prompt(contradictions: list[dict]) -> str:
    if not contradictions:
        return "None detected."
    lines = []
    for i, c in enumerate(contradictions, 1):
        lines.append(
            f"{i}. Topic: {c['topic']}\n"
            f"   Source A ({c['source_a']}): {c['claim_a']}\n"
            f"   Source B ({c['source_b']}): {c['claim_b']}"
        )
    return "\n".join(lines)



def _clean_source_for_writer(text: str, max_chars: int = 800) -> str:
    """
    Strip the Exa-Score line and truncate to max_chars.
    800 chars * 9 sources = ~7200 chars, leaving room for the prompt itself.
    """
    lines = [l for l in text.splitlines() if not l.startswith("Exa-Score:")]
    cleaned = "\n".join(lines).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "... [truncated]"
    return cleaned

def writer_agent(state: ResearchState) -> ResearchState:
    validated = state.get("validated_results", [])
    contradictions = state.get("contradictions", [])
    query = state["query"]

    if not validated:
        return {"report": "No sufficient sources were found to generate a report for this query.", "citations": []}

    numbered_sources = "\n\n---\n\n".join(
        f"[SOURCE {i+1}]\n{_clean_source_for_writer(r)}" for i, r in enumerate(validated)
    )
    contradictions_block = _format_contradictions_for_prompt(contradictions)

    prompt = f"""You are an expert research analyst. Write a comprehensive, well-structured research report answering the query below.

QUERY: "{query}"

VALIDATED SOURCES (pre-filtered for quality and ranked by score):
{numbered_sources}

DETECTED CONTRADICTIONS BETWEEN SOURCES:
{contradictions_block}

REPORT FORMAT (use Markdown):
# [Title]

## Executive Summary
2-3 sentences capturing the single most important answer to the query.

## Findings
Write in connected prose paragraphs - NO bullet lists. For every factual claim, cite the source inline like [Source 1] or [Source 1, 3] for multiple. Depth over breadth: one well-cited paragraph beats five vague ones.

## Conflicting Evidence
ONLY include this section if contradictions were detected above. Present each conflict fairly - show both sides, do not pick one. Explain why the conflict matters to the reader.

## Research Gaps
1-2 sentences on what these sources do NOT answer and what a follow-up search should target.

## Sources
List each [Source N] you cited as:
N. [URL]

RULES:
- Never invent facts not in the sources.
- Never skip a contradiction that was flagged - it must appear in the report.
- Inline citations must match the numbered sources above exactly.
- Do NOT use bullet points in Findings."""

    report_text = _invoke_with_fallback(prompt, max_tokens=4096).strip()

    used_indices = sorted(set(
        int(n) - 1
        for n in re.findall(r'\[Source\s+(\d+)', report_text)
    ))
    citations = []
    for idx in used_indices:
        if idx < len(validated):
            urls = _extract_urls(validated[idx])
            citations.append(urls[0] if urls else f"Source {idx+1} (no URL)")

    return {"report": report_text, "citations": citations}