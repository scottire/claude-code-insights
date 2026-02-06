"""
Core Pipeline Module for Claude Code Insights.

This module contains the main pipeline logic extracted from cli.py,
designed to be importable without CLI dependencies.

Key differences from cli.py:
- No module-level weave.init() - call init_weave() explicitly
- Explicit api_key parameter instead of environment variable
- on_progress callback instead of rich.Progress
- Plain functions instead of Typer decorators
"""

import json
import os
import re
from concurrent.futures import as_completed
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Callable, Literal

import weave
from openai import OpenAI
from weave import Content

from insights.prompts import get_prompt

# ============================================================================
# Weave Initialization
# ============================================================================

_weave_initialized = False
_weave_url = None


def init_weave(project: str = "claude-code-insights") -> str:
    """
    Initialize Weave tracing.

    Call this explicitly before running the pipeline if you want tracing.
    This is NOT called on import to avoid side effects.

    Returns:
        The Weave project URL (e.g. https://wandb.ai/entity/project/weave)
    """
    global _weave_initialized, _weave_url
    if not _weave_initialized:
        client = weave.init(project)
        _weave_url = f"https://wandb.ai/{client.entity}/{client.project}/weave"
        _weave_initialized = True
    return _weave_url


def get_weave_url() -> str | None:
    """Get the Weave URL if initialized."""
    return _weave_url


# ============================================================================
# LLM Client Setup
# ============================================================================

MODEL = "moonshotai/Kimi-K2-Instruct-0905"


def get_client(api_key: str | None = None) -> OpenAI:
    """
    Get OpenAI client configured for W&B inference.

    Args:
        api_key: W&B API key. If not provided, falls back to WANDB_API_KEY env var.
    """
    key = api_key or os.environ.get("WANDB_API_KEY")
    if not key:
        raise ValueError("API key required: pass api_key parameter or set WANDB_API_KEY")
    return OpenAI(
        base_url='https://api.inference.wandb.ai/v1',
        api_key=key
    )


def call_llm(prompt: str, system: str = "", api_key: str | None = None) -> dict:
    """Call LLM and parse JSON response."""
    client = get_client(api_key)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content

    # Try to parse JSON, with fallback cleanup for common issues
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try cleaning control characters
        cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try extracting JSON from markdown code blocks
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', cleaned)
            if match:
                return json.loads(match.group(1))
            raise


# ============================================================================
# Parallel Execution Utility
# ============================================================================

def parallel_llm_map(
    items: list,
    worker_fn: Callable,
    max_workers: int = 10,
    on_progress: Callable[[int, int], None] | None = None
) -> tuple[list, list]:
    """
    Execute LLM calls in parallel.

    Args:
        items: List of items to process
        worker_fn: Function that takes an item and returns a result (or raises)
        max_workers: Number of parallel workers
        on_progress: Optional callback(completed, total) for progress updates

    Returns:
        Tuple of (results, errors) where errors contain {"error": str, "item": item}
    """
    results = []
    errors = []
    completed = 0
    total = len(items)

    with weave.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker_fn, item): item for item in items}

        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                errors.append({"error": str(e), "item": item})

            completed += 1
            if on_progress:
                on_progress(completed, total)

    return results, errors


# ============================================================================
# Session Filtering and Parsing
# ============================================================================

FACET_SCHEMA = {
    "underlying_goal": "What the user fundamentally wanted to achieve",
    "goal_categories": {"category_name": "count"},
    "outcome": "fully_achieved|mostly_achieved|partially_achieved|not_achieved|unclear_from_transcript",
    "user_satisfaction_counts": {"level": "count"},
    "claude_helpfulness": "unhelpful|slightly_helpful|moderately_helpful|very_helpful|essential",
    "session_type": "single_task|multi_task|iterative_refinement|exploration|quick_question",
    "friction_counts": {"friction_type": "count"},
    "friction_detail": "One sentence describing friction or empty",
    "primary_success": "none|fast_accurate_search|correct_code_edits|good_explanations|proactive_help|multi_file_changes|good_debugging",
    "brief_summary": "One sentence: what user wanted and whether they got it",
    "session_id": "session_uuid"
}

FRICTION_CATEGORIES = [
    "wrong_approach", "misunderstood_request", "buggy_code", "user_rejected_action",
    "claude_got_blocked", "user_stopped_early", "wrong_file_or_location",
    "excessive_changes", "slow_or_verbose", "tool_failed", "user_unclear", "external_issue"
]

ANALYSIS_PROMPTS = [
    "project_areas",
    "interaction_style",
    "what_works",
    "friction_analysis",
    "suggestions",
    "on_the_horizon",
    "fun_ending"
]


def parse_jsonl_session(jsonl_path: Path) -> dict:
    """Parse a Claude Code JSONL session file into a transcript dict."""
    messages = []
    session_id = None
    start_time = None
    end_time = None
    cwd = None

    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type")

            # Extract session metadata from user/assistant messages
            if event_type in ("user", "assistant"):
                if not session_id:
                    session_id = event.get("sessionId")
                if not cwd:
                    cwd = event.get("cwd")

                timestamp = event.get("timestamp")
                if timestamp:
                    if not start_time or timestamp < start_time:
                        start_time = timestamp
                    if not end_time or timestamp > end_time:
                        end_time = timestamp

                msg = event.get("message", {})
                content = msg.get("content", "")

                # Handle content that might be a list of content blocks
                if isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                            elif block.get("type") == "tool_use":
                                text_parts.append(f"[Tool: {block.get('name', 'unknown')}]")
                            elif block.get("type") == "tool_result":
                                text_parts.append(f"[Tool result]")
                        elif isinstance(block, str):
                            text_parts.append(block)
                    content = "\n".join(text_parts)

                if content:
                    messages.append({
                        "role": msg.get("role", event_type),
                        "content": content,
                        "timestamp": timestamp
                    })

    return {
        "session_id": session_id or jsonl_path.stem,
        "messages": messages,
        "start_time": start_time,
        "end_time": end_time,
        "cwd": cwd
    }


def load_transcripts_from_claude_dir(claude_dir: Path, limit: int | None = None) -> list[dict]:
    """Load transcripts from ~/.claude/projects directory."""
    transcripts = []

    # Find all JSONL files (excluding subagents)
    jsonl_files = []
    for project_dir in claude_dir.glob("*"):
        if project_dir.is_dir():
            for jsonl_file in project_dir.glob("*.jsonl"):
                # Skip subagent files
                if "subagents" not in str(jsonl_file):
                    jsonl_files.append(jsonl_file)

    # Sort by modification time (most recent first)
    jsonl_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    # Load until we have `limit` valid sessions (not just first `limit` files)
    for jsonl_file in jsonl_files:
        if limit and len(transcripts) >= limit:
            break
        try:
            transcript = parse_jsonl_session(jsonl_file)
            if transcript["messages"]:  # Only include non-empty sessions
                transcripts.append(transcript)
        except Exception:
            pass  # Skip files that can't be parsed

    return transcripts


def should_skip_session(transcript: dict) -> bool:
    """Check if session should be skipped (warmup, internal, too short)."""
    messages = transcript.get("messages", [])
    user_messages = [m for m in messages if m.get("role") == "user"]

    # Skip if fewer than 2 user messages
    if len(user_messages) < 2:
        return True

    # Skip warmup_minimal sessions
    if transcript.get("type") == "warmup_minimal":
        return True

    # Skip sessions shorter than 1 minute
    start_time = transcript.get("start_time")
    end_time = transcript.get("end_time")
    if start_time and end_time:
        try:
            start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            end = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
            duration_minutes = (end - start).total_seconds() / 60
            if duration_minutes < 1:
                return True
        except (ValueError, TypeError):
            pass

    return False


def chunk_transcript(transcript: str, max_size: int = 30000) -> list[str]:
    """Split large transcripts into chunks for summarization."""
    if len(transcript) <= max_size:
        return [transcript]

    chunks = []
    while transcript:
        chunk = transcript[:max_size]
        # Try to break at a newline
        last_newline = chunk.rfind("\n")
        if last_newline > max_size // 2:
            chunk = transcript[:last_newline]
        chunks.append(chunk)
        transcript = transcript[len(chunk):]
    return chunks


# ============================================================================
# Stage 1: Facet Extraction
# ============================================================================

@weave.op(color="yellow")
def summarize_chunk(chunk: str, api_key: str | None = None) -> str:
    """Summarize a transcript chunk using LLM."""
    if len(chunk.strip()) < 100:
        return ""
    prompt = get_prompt("summary") + chunk
    result = call_llm(prompt, api_key=api_key)
    return result.get("summary", "")


@weave.op(color="blue")
def extract_facets_from_transcript(transcript: dict, api_key: str | None = None) -> dict | None:
    """Extract facets from a single session transcript."""
    if should_skip_session(transcript):
        return None

    # Convert transcript to string
    transcript_str = json.dumps(transcript, indent=2)

    # Handle large transcripts
    if len(transcript_str) > 30000:
        chunks = chunk_transcript(transcript_str)
        chunks = [c for c in chunks if len(c.strip()) >= 100]
        summaries = [summarize_chunk(chunk, api_key) for chunk in chunks if chunk.strip()]
        transcript_str = "\n\n---\n\n".join(summaries)

    # Build the prompt
    prompt = get_prompt("facet_extraction") + transcript_str
    schema_str = json.dumps(FACET_SCHEMA, indent=2)

    full_prompt = f"""
{prompt}

RESPOND WITH ONLY A VALID JSON OBJECT matching this schema:
{schema_str}

Friction categories to use: {', '.join(FRICTION_CATEGORIES)}
"""

    result = call_llm(full_prompt, api_key=api_key)
    result["session_id"] = transcript.get("session_id", transcript.get("id", "unknown"))
    result["start_time"] = transcript.get("start_time")
    result["end_time"] = transcript.get("end_time")
    return result


# ============================================================================
# Stage 2: Aggregation (Pure Python - No LLM)
# ============================================================================

def merge_count_dicts(target: dict, source: dict) -> dict:
    """Merge two count dictionaries."""
    for key, count in source.items():
        try:
            count_int = int(count) if isinstance(count, str) else count
        except (ValueError, TypeError):
            count_int = 1
        target[key] = target.get(key, 0) + count_int
    return target


def aggregate_facets(facets: list[dict]) -> dict:
    """Aggregate facets into statistics."""
    aggregated = {
        "total_sessions": len(facets),
        "sessions_with_facets": len([f for f in facets if f]),
        "goal_categories": {},
        "outcomes": {},
        "satisfaction": {},
        "helpfulness": {},
        "session_types": {},
        "friction": {},
        "primary_successes": {},
        "tool_counts": {},
        "languages": {},
        "projects": {},
        "summaries": [],
        "date_range": {"start": None, "end": None},
        "days_active": 0,
    }

    all_dates = []

    for facet in facets:
        if not facet:
            continue

        if facet.get("start_time"):
            all_dates.append(facet["start_time"])
        if facet.get("end_time"):
            all_dates.append(facet["end_time"])

        merge_count_dicts(aggregated["goal_categories"], facet.get("goal_categories", {}))

        outcome = facet.get("outcome", "unknown")
        aggregated["outcomes"][outcome] = aggregated["outcomes"].get(outcome, 0) + 1

        merge_count_dicts(aggregated["satisfaction"], facet.get("user_satisfaction_counts", {}))

        helpfulness = facet.get("claude_helpfulness", "unknown")
        aggregated["helpfulness"][helpfulness] = aggregated["helpfulness"].get(helpfulness, 0) + 1

        session_type = facet.get("session_type", "unknown")
        aggregated["session_types"][session_type] = aggregated["session_types"].get(session_type, 0) + 1

        merge_count_dicts(aggregated["friction"], facet.get("friction_counts", {}))

        success = facet.get("primary_success", "none")
        aggregated["primary_successes"][success] = aggregated["primary_successes"].get(success, 0) + 1

        if facet.get("brief_summary"):
            aggregated["summaries"].append({
                "session_id": facet.get("session_id"),
                "summary": facet.get("brief_summary"),
                "underlying_goal": facet.get("underlying_goal", ""),
                "friction_detail": facet.get("friction_detail", "")
            })

    aggregated["top_goals"] = sorted(
        aggregated["goal_categories"].items(),
        key=lambda x: x[1],
        reverse=True
    )[:8]

    aggregated["top_friction"] = sorted(
        aggregated["friction"].items(),
        key=lambda x: x[1],
        reverse=True
    )[:8]

    if all_dates:
        sorted_dates = sorted(all_dates)
        start_date = sorted_dates[0][:10] if sorted_dates[0] else None
        end_date = sorted_dates[-1][:10] if sorted_dates[-1] else None
        aggregated["date_range"] = {"start": start_date, "end": end_date}

        unique_dates = set()
        for d in all_dates:
            if d:
                unique_dates.add(d[:10])
        aggregated["days_active"] = len(unique_dates)

    return aggregated


# ============================================================================
# Stage 3: Analysis Prompts
# ============================================================================

@weave.op(
    color="green",
    call_display_name=lambda call: f"analyze:{call.inputs['prompt_name']}"
)
def run_analysis_prompt(
    prompt_name: str,
    aggregated: dict,
    facets: list[dict],
    api_key: str | None = None
) -> tuple[str, dict]:
    """Run a single analysis prompt and return (name, result)."""
    prompt_template = get_prompt(prompt_name)

    context = f"""
AGGREGATED DATA:
{json.dumps(aggregated, indent=2)}

SESSION SUMMARIES:
{json.dumps(aggregated.get('summaries', []), indent=2)}
"""

    full_prompt = prompt_template + "\n\nDATA:\n" + context
    result = call_llm(full_prompt, api_key=api_key)
    return prompt_name, result


# ============================================================================
# Stage 4: At-a-Glance Synthesis
# ============================================================================

@weave.op(color="orange")
def generate_at_a_glance(analysis: dict, aggregated: dict, api_key: str | None = None) -> dict:
    """Generate at-a-glance synthesis from analysis results."""
    prompt_template = get_prompt("at_a_glance")

    prompt = prompt_template.replace("${aggregated_data}", json.dumps(aggregated, indent=2))
    prompt = prompt.replace("${project_areas}", json.dumps(analysis.get("project_areas", {}), indent=2))
    prompt = prompt.replace("${what_works}", json.dumps(analysis.get("what_works", {}), indent=2))
    prompt = prompt.replace("${friction_analysis}", json.dumps(analysis.get("friction_analysis", {}), indent=2))

    suggestions = analysis.get("suggestions", {})
    prompt = prompt.replace("${suggestions.features_to_try}", json.dumps(suggestions.get("features_to_try", []), indent=2))
    prompt = prompt.replace("${suggestions.usage_patterns}", json.dumps(suggestions.get("usage_patterns", []), indent=2))
    prompt = prompt.replace("${on_the_horizon}", json.dumps(analysis.get("on_the_horizon", {}), indent=2))

    return call_llm(prompt, api_key=api_key)


# ============================================================================
# Full Pipeline
# ============================================================================

def _redact_transcripts(inputs: dict) -> dict:
    """Redact transcripts from traced inputs to keep traces clean."""
    return {k: v for k, v in inputs.items() if k != "transcripts"}


@weave.op(
    kind="agent",
    color="purple",
    postprocess_inputs=_redact_transcripts
)
def run_insights_pipeline(
    transcripts: list[dict],
    api_key: str | None = None,
    workers: int = 5,
    log: Callable[[str], None] | None = None
) -> Annotated[bytes, Content[Literal['html']]]:
    """
    Run the full insights pipeline on transcripts.

    Args:
        transcripts: List of parsed session transcripts
        api_key: W&B API key (or set WANDB_API_KEY env var)
        workers: Number of parallel workers
        log: Optional callback for log messages (e.g. print or mo.output.append)

    Returns:
        HTML report bytes
    """
    from insights.report_template import generate_html_report

    def emit(msg: str):
        if log:
            log(msg)

    # Filter valid transcripts
    valid_transcripts = [t for t in transcripts if not should_skip_session(t)]
    skipped = len(transcripts) - len(valid_transcripts)
    emit(f"Processing {len(valid_transcripts)} sessions ({skipped} skipped)")

    # Stage 1: Extract facets in parallel
    emit(f"\n[Stage 1] Extracting facets with {workers} workers...")

    def extract_worker(t):
        return extract_facets_from_transcript(t, api_key)

    facets, errors = parallel_llm_map(
        items=valid_transcripts,
        worker_fn=extract_worker,
        max_workers=workers,
        on_progress=lambda c, t: emit(f"  Facets: {c}/{t}")
    )
    emit(f"  Extracted {len(facets)} facets ({len(errors)} errors)")

    # Stage 2: Aggregate (pure Python)
    emit(f"\n[Stage 2] Aggregating facets...")
    aggregated = aggregate_facets(facets)
    emit(f"  Aggregated {aggregated['total_sessions']} sessions")

    # Stage 3: Run analysis prompts in parallel
    emit(f"\n[Stage 3] Running 7 analysis prompts in parallel...")

    def analysis_worker(prompt_name: str) -> tuple[str, dict]:
        return run_analysis_prompt(prompt_name, aggregated, facets, api_key)

    raw_results, analysis_errors = parallel_llm_map(
        items=ANALYSIS_PROMPTS,
        worker_fn=analysis_worker,
        max_workers=7,
        on_progress=lambda c, t: emit(f"  Analysis: {c}/{t}")
    )

    analysis = {name: result for name, result in raw_results}
    for err in analysis_errors:
        analysis[err["item"]] = {"error": err["error"]}
    emit(f"  Completed: {', '.join(analysis.keys())}")

    # Stage 4: Generate at-a-glance synthesis
    emit(f"\n[Stage 4] Generating at-a-glance synthesis...")
    at_a_glance = generate_at_a_glance(analysis, aggregated, api_key)
    emit(f"  Done!")

    # Calculate date range
    all_dates = []
    for t in transcripts:
        if t.get("start_time"):
            all_dates.append(t["start_time"][:10])
        if t.get("end_time"):
            all_dates.append(t["end_time"][:10])

    if all_dates:
        sorted_dates = sorted(set(all_dates))
        date_range = f"{sorted_dates[0]} to {sorted_dates[-1]}"
    else:
        date_range = "unknown"

    # Build final report
    report = {
        "at_a_glance": at_a_glance,
        "analysis": analysis,
        "aggregated": aggregated,
        "generated_at": datetime.now().isoformat(),
        "session_count": len(transcripts),
        "date_range": date_range
    }

    # Generate HTML report
    emit(f"\nGenerating HTML report...")
    html = generate_html_report(report)
    emit(f"Pipeline complete! Report size: {len(html) / 1024:.1f} KB")

    return html.encode('utf-8')


def load_transcripts(
    transcripts_path: Path,
    use_claude_dir: bool = False,
    limit: int | None = None
) -> tuple[list[dict], str]:
    """
    Load transcripts from various sources.

    Args:
        transcripts_path: Path to transcripts (directory or file)
        use_claude_dir: Treat as ~/.claude/projects directory
        limit: Limit number of sessions to load

    Returns:
        Tuple of (transcripts, date_range_str)
    """
    transcripts = []

    if use_claude_dir or (transcripts_path.is_dir() and (transcripts_path / "projects").exists()):
        projects_dir = transcripts_path / "projects" if (transcripts_path / "projects").exists() else transcripts_path
        transcripts = load_transcripts_from_claude_dir(projects_dir, limit=limit)
    elif transcripts_path.is_dir():
        jsonl_files = list(transcripts_path.glob("*.jsonl"))
        if jsonl_files:
            for f in jsonl_files[:limit] if limit else jsonl_files:
                try:
                    transcripts.append(parse_jsonl_session(f))
                except Exception:
                    pass
        else:
            json_files = list(transcripts_path.glob("*.json"))
            for f in json_files[:limit] if limit else json_files:
                transcripts.append(json.loads(f.read_text()))
    elif transcripts_path.suffix == ".jsonl":
        transcripts = [parse_jsonl_session(transcripts_path)]
    else:
        data = json.loads(transcripts_path.read_text())
        if isinstance(data, list):
            transcripts = data[:limit] if limit else data
        else:
            transcripts = [data]

    # Calculate date range
    all_dates = []
    for t in transcripts:
        if t.get("start_time"):
            all_dates.append(t["start_time"][:10])
        if t.get("end_time"):
            all_dates.append(t["end_time"][:10])

    if all_dates:
        sorted_dates = sorted(set(all_dates))
        date_range = f"{sorted_dates[0]} to {sorted_dates[-1]}"
    else:
        date_range = "unknown"

    return transcripts, date_range
