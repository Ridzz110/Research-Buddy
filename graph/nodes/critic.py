from graph.state import ResearchState, CriticReport, ScoredSource, Contradiction
from urllib.parse import urlparse
import logging
from graph.nodes.shared_utils import _extract_exa_score, _extract_urls, _is_homepage, _invoke_with_fallback, _parse_llm_json
import json
logger = logging.getLogger(__name__)

# NODE 3 — CRITIC

def _truncate_for_critic(text: str, max_chars: int = 500) -> str:
    lines = text.splitlines()
    title = lines[0] if lines else ""
    meta = [l for l in lines if l.startswith("Source:") or l.startswith("Exa-Score:")]
    content_lines = [l for l in lines if not l.startswith("Source:") and not l.startswith("Exa-Score:") and l != title]
    content_preview = " ".join(content_lines)[:max_chars]
    return "\n".join(filter(None, [title, content_preview, *meta]))


def _build_critic_prompt(query: str, results: list[str]) -> str:
    numbered = "\n\n".join(f"[SOURCE {i+1}]\n{_truncate_for_critic(r)}" for i, r in enumerate(results))
    return f"""You are a rigorous research critic. Evaluate each source for the query below and return a JSON assessment.

QUERY: "{query}"

SOURCES:
{numbered}

SCORING RULES:
- relevance_score (0.0-1.0): how directly does the source answer the query?
- quality_score (0.0-1.0): judge from the CONTENT ITSELF:
    High (0.8-1.0): cites data/statistics, names authors or institutions,
                    references methodology, peer-reviewed, government/IGO source
    Medium (0.5-0.7): informed opinion, industry report, established news outlet
    Low (0.0-0.4): no evidence cited, vague generalities, promotional tone,
                   anonymous author, no publication date
- kept: true = use in report, false = discard
  DROP if: off-topic, pure marketing fluff, same domain as a higher-scored source
           (keep only the best result per root domain), no concrete information
  KEEP if: adds unique factual value

Also detect CONTRADICTIONS: if two sources make opposing factual claims on the same topic, flag them.

Return ONLY valid JSON - no markdown fences, no explanation outside the JSON.

{{
  "assessments": [
    {{
      "source_index": 1,
      "relevance_score": 0.85,
      "quality_score": 0.80,
      "kept": true,
      "reason": "Peer-reviewed study with methodology and citation data directly addressing the query."
    }}
  ],
  "contradictions": [
    {{
      "topic": "short topic label",
      "claim_a": "what source X claims",
      "source_a_index": 2,
      "claim_b": "what source Y claims",
      "source_b_index": 5
    }}
  ]
}}"""


def critic_agent(state: ResearchState) -> ResearchState:
    results = state.get("search_results", [])
    query = state["query"]

    if not results:
        logger.warning("critic_agent: no search results to evaluate.")
        return {
            "validated_results": [],
            "dropped_results": [],
            "contradictions": [],
            "critic_report": CriticReport(total_input=0),
        }

    # ── Pre-filter: drop homepages before the LLM sees them ──────────────
    clean_results: list[str] = []
    pre_dropped: list[ScoredSource] = []

    for r in results:
        urls = _extract_urls(r)
        url = urls[0] if urls else ""
        if _is_homepage(url):
            logger.info(f"critic_agent: dropped homepage {url}")
            pre_dropped.append(
                ScoredSource(content=r, url=url, kept=False,
                             reason="Homepage URL - no specific content to cite")
            )
        else:
            clean_results.append(r)

    results = clean_results

    if not results:
        return {
            "validated_results": [],
            "dropped_results": [s.content for s in pre_dropped],
            "contradictions": [],
            "critic_report": CriticReport(
                dropped=pre_dropped,
                total_input=len(state.get("search_results", [])),
                total_dropped=len(pre_dropped),
            ),
        }

    prompt = _build_critic_prompt(query, results)

    try:
        raw = _invoke_with_fallback(prompt, max_tokens=1024)
        parsed = _parse_llm_json(raw)
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"critic_agent: JSON parse failed ({e}). Keeping all results.")
        return {
            "validated_results": results,
            "dropped_results": [s.content for s in pre_dropped],
            "contradictions": [],
            "critic_report": CriticReport(
                kept=[ScoredSource(content=r, reason="fallback: parse error") for r in results],
                dropped=pre_dropped,
                total_input=len(results),
                total_kept=len(results),
            ),
        }

    assessments = parsed.get("assessments", [])
    raw_contradictions = parsed.get("contradictions", [])

    # ── Build ScoredSource per result ────────────────────────────────────
    scored: list[ScoredSource] = []
    assessed_indices = set()

    for item in assessments:
        idx = item.get("source_index", 1) - 1
        if idx < 0 or idx >= len(results):
            continue
        assessed_indices.add(idx)

        content = results[idx]
        urls = _extract_urls(content)
        url = urls[0] if urls else ""
        domain = urlparse(url).netloc.replace("www.", "") if url else ""

        relevance = float(item.get("relevance_score", 0.5))
        quality = float(item.get("quality_score", 0.5))
        exa_score = _extract_exa_score(content)

        # Composite: 35% LLM relevance + 35% LLM quality + 30% Exa neural score
        composite = round(0.35 * relevance + 0.35 * quality + 0.30 * exa_score, 3)

        scored.append(ScoredSource(
            content=content,
            url=url,
            domain=domain,
            relevance_score=relevance,
            quality_score=quality,
            composite_score=composite,
            kept=bool(item.get("kept", True)),
            reason=item.get("reason", ""),
        ))

    # Any result the LLM skipped -> keep with Exa score only
    for i, r in enumerate(results):
        if i not in assessed_indices:
            exa_score = _extract_exa_score(r)
            scored.append(ScoredSource(
                content=r,
                composite_score=round(0.5 + 0.30 * exa_score, 3),
                kept=True,
                reason="not assessed by LLM - kept with Exa score only"
            ))

    # ── Build Contradiction objects ───────────────────────────────────────
    contradictions: list[Contradiction] = []
    for c in raw_contradictions:
        a_idx = c.get("source_a_index", 1) - 1
        b_idx = c.get("source_b_index", 1) - 1
        a_url = (_extract_urls(results[a_idx]) or [f"Source {a_idx+1}"])[0] if a_idx < len(results) else f"Source {a_idx+1}"
        b_url = (_extract_urls(results[b_idx]) or [f"Source {b_idx+1}"])[0] if b_idx < len(results) else f"Source {b_idx+1}"
        contradictions.append(Contradiction(
            topic=c.get("topic", ""),
            claim_a=c.get("claim_a", ""),
            source_a=a_url,
            claim_b=c.get("claim_b", ""),
            source_b=b_url,
        ))

    kept = sorted([s for s in scored if s.kept], key=lambda s: s.composite_score, reverse=True)
    dropped = [s for s in scored if not s.kept] + pre_dropped

    report = CriticReport(
        kept=kept,
        dropped=dropped,
        contradictions=contradictions,
        total_input=len(state.get("search_results", [])),
        total_kept=len(kept),
        total_dropped=len(dropped),
    )

    logger.info(
        f"critic_agent: {report.total_input} in -> {len(kept)} kept, "
        f"{len(dropped)} dropped, {len(contradictions)} contradictions."
    )

    return {
        "validated_results": [s.content for s in kept],
        "dropped_results": [s.content for s in dropped],
        "contradictions": [
            {
                "topic": c.topic,
                "claim_a": c.claim_a,
                "source_a": c.source_a,
                "claim_b": c.claim_b,
                "source_b": c.source_b,
            }
            for c in contradictions
        ],
        "critic_report": report,
    }
