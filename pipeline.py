"""
Claude Code Insights Pipeline.

This replicates the /insights command from Claude Code:
1. Extract facets from session transcripts
2. Aggregate facets into statistics
3. Run 7 analysis prompts in parallel (map-reduce)
4. Synthesize at-a-glance summary

Usage:
    from pipeline import run_insights_pipeline, load_transcripts, parse_jsonl_session

    transcripts, date_range = load_transcripts(Path.home() / ".claude" / "projects", limit=10)
    html_bytes = run_insights_pipeline(transcripts, len(transcripts), date_range)
"""

import json
import os
import re
from concurrent.futures import as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable

import weave
from openai import OpenAI
from weave import Content

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
# LLM Client Setup
# ============================================================================

def get_client(api_key: str | None = None) -> OpenAI:
    """
    Get OpenAI client configured for W&B inference.

    Args:
        api_key: W&B API key. If not provided, falls back to WANDB_API_KEY env var.
    """
    key = api_key or os.environ.get("WANDB_API_KEY")
    if not key:
        raise ValueError("API key required: pass api_key parameter or set WANDB_API_KEY env var")
    return OpenAI(
        base_url='https://api.inference.wandb.ai/v1',
        api_key=key
    )

MODEL = "zai-org/GLM-4.5"

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
        import re
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
# Prompt Loading
# ============================================================================

PROMPTS_DIR = Path(__file__).parent / "prompts"

def load_prompt(name: str) -> str:
    """Load a prompt from the prompts directory."""
    return (PROMPTS_DIR / f"{name}.md").read_text()


# ============================================================================
# Stage 1: Facet Extraction
# ============================================================================

# Fixed goal categories (from real CLI)
GOAL_CATEGORIES = [
    "debug_investigate", "implement_feature", "fix_bug", "write_script_tool",
    "refactor_code", "configure_system", "create_pr_commit", "analyze_data",
    "understand_codebase", "write_tests", "write_docs", "deploy_infra", "warmup_minimal"
]

# Fixed friction categories (from real CLI)
FRICTION_CATEGORIES = [
    "misunderstood_request", "wrong_approach", "buggy_code", "user_rejected_action",
    "claude_got_blocked", "user_stopped_early", "wrong_file_or_location",
    "excessive_changes", "slow_or_verbose", "tool_failed", "user_unclear", "external_issue"
]

# Fixed success categories (from real CLI)
SUCCESS_CATEGORIES = [
    "none", "fast_accurate_search", "correct_code_edits", "good_explanations",
    "proactive_help", "multi_file_changes", "good_debugging"
]

FACET_SCHEMA = {
    "underlying_goal": "What the user fundamentally wanted to achieve",
    "goal_categories": {f"ONLY use: {', '.join(GOAL_CATEGORIES)}": "count"},
    "outcome": "fully_achieved|mostly_achieved|partially_achieved|not_achieved|unclear_from_transcript",
    "user_satisfaction_counts": {"frustrated|dissatisfied|likely_satisfied|satisfied|happy": "count"},
    "claude_helpfulness": "unhelpful|slightly_helpful|moderately_helpful|very_helpful|essential",
    "session_type": "single_task|multi_task|iterative_refinement|exploration|quick_question",
    "friction_counts": {f"ONLY use: {', '.join(FRICTION_CATEGORIES)}": "count"},
    "friction_detail": "One sentence describing friction or empty",
    "primary_success": f"ONLY use: {', '.join(SUCCESS_CATEGORIES)}",
    "brief_summary": "One sentence: what user wanted and whether they got it",
    "user_instructions_to_claude": ["List of explicit instructions user gave Claude, e.g. 'always run tests', 'use TypeScript'"],
    "session_id": "session_uuid"
}



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


def load_transcripts_from_claude_dir(claude_dir: Path, limit: int = None) -> list[dict]:
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

    if limit:
        jsonl_files = jsonl_files[:limit]

    for jsonl_file in jsonl_files:
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


@weave.op(color="yellow")
def summarize_chunk(chunk: str, api_key: str | None = None) -> str:
    """Summarize a transcript chunk using LLM."""
    # Skip chunks that are too small to be meaningful
    if len(chunk.strip()) < 100:
        return ""
    prompt = load_prompt("summary") + chunk
    result = call_llm(prompt, api_key=api_key)
    return result.get("summary", "")


def format_transcript_compact(transcript: dict) -> str:
    """Format transcript in compact style matching real Claude CLI.

    Output format:
    [Tool: Read]
    [Assistant]: Now I'll implement...
    [Tool: Edit]
    [User]: looks good
    """
    lines = []
    for msg in transcript.get("messages", []):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if role == "assistant":
            # For assistant, prefix text with [Assistant]: and keep [Tool: X] on own lines
            parts = content.split("\n")
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if part.startswith("[Tool:"):
                    lines.append(part)
                elif part == "[Tool result]":
                    continue  # Skip tool results
                else:
                    lines.append(f"[Assistant]: {part}")
        elif role == "user":
            # User messages - just show the text
            text = content.strip()
            if text and not text.startswith("[Tool"):
                lines.append(f"[User]: {text[:500]}")  # Truncate long user messages

    return "\n".join(lines)


@weave.op(color="blue")
def extract_facets_from_transcript(transcript: dict, api_key: str | None = None) -> dict | None:
    """Extract facets from a single session transcript."""
    if should_skip_session(transcript):
        return None

    # Convert transcript to compact string format (like real CLI)
    transcript_str = format_transcript_compact(transcript)

    # Handle large transcripts
    if len(transcript_str) > 30000:
        chunks = chunk_transcript(transcript_str)
        # Filter out chunks that are too small to be meaningful (< 100 chars)
        chunks = [c for c in chunks if len(c.strip()) >= 100]
        summaries = [summarize_chunk(chunk, api_key) for chunk in chunks if chunk.strip()]
        transcript_str = "\n\n---\n\n".join(summaries)

    # Build the prompt
    prompt = load_prompt("facet_extraction") + transcript_str
    schema_str = json.dumps(FACET_SCHEMA, indent=2)

    full_prompt = f"""
{prompt}

RESPOND WITH ONLY A VALID JSON OBJECT matching this schema:
{schema_str}
"""

    result = call_llm(full_prompt, api_key=api_key)
    result["session_id"] = transcript.get("session_id", transcript.get("id", "unknown"))
    # Include timestamps for date range calculation
    result["start_time"] = transcript.get("start_time")
    result["end_time"] = transcript.get("end_time")
    return result




# ============================================================================
# Stage 2: Aggregation (Pure Python - No LLM)
# ============================================================================

RESPONSE_TIME_BUCKETS = ["2-10s", "10-30s", "30s-1m", "1-2m", "2-5m", "5-15m", ">15m"]

TIME_PERIODS = {
    "Morning": range(6, 12),
    "Afternoon": range(12, 18),
    "Evening": range(18, 24),
    "Night": range(0, 6)
}


def merge_count_dicts(target: dict, source: dict) -> dict:
    """Merge two count dictionaries."""
    for key, count in source.items():
        # Handle counts that might be strings
        try:
            count_int = int(count) if isinstance(count, str) else count
        except (ValueError, TypeError):
            count_int = 1  # Default to 1 if can't parse
        target[key] = target.get(key, 0) + count_int
    return target


def bucket_response_time(seconds: float) -> str:
    """Categorize response time into bucket."""
    if seconds < 2:
        return "2-10s"  # Group with 2-10s
    elif seconds < 10:
        return "2-10s"
    elif seconds < 30:
        return "10-30s"
    elif seconds < 60:
        return "30s-1m"
    elif seconds < 120:
        return "1-2m"
    elif seconds < 300:
        return "2-5m"
    elif seconds < 900:
        return "5-15m"
    else:
        return ">15m"


def get_hour_period(hour: int) -> str:
    """Get time period for an hour."""
    for period, hours in TIME_PERIODS.items():
        if hour in hours:
            return period
    return "Unknown"


# Language detection by file extension (from real CLI)
LANGUAGE_MAP = {
    ".ts": "TypeScript", ".tsx": "TypeScript",
    ".js": "JavaScript", ".jsx": "JavaScript",
    ".py": "Python", ".rb": "Ruby", ".go": "Go", ".rs": "Rust",
    ".java": "Java", ".md": "Markdown", ".json": "JSON",
    ".yaml": "YAML", ".yml": "YAML", ".sh": "Shell",
    ".css": "CSS", ".html": "HTML",
}


def aggregate_facets_impl(facets: list[dict], transcripts: list[dict] = None) -> dict:
    """Aggregate facets and session-level stats into statistics."""
    transcripts = transcripts or []

    aggregated = {
        "total_sessions": len(transcripts) if transcripts else len(facets),
        "sessions_with_facets": len([f for f in facets if f]),
        "total_messages": 0,
        "total_duration_hours": 0,
        "git_commits": 0,
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
        "response_time_buckets": {b: 0 for b in RESPONSE_TIME_BUCKETS},
        "hour_histogram": {p: 0 for p in TIME_PERIODS.keys()},
        "summaries": [],
        "date_range": {"start": None, "end": None},
        "days_active": 0,
    }

    all_dates = []

    # Extract session-level stats from raw transcripts
    for transcript in transcripts:
        messages = transcript.get("messages", [])

        # Count user messages
        user_messages = [m for m in messages if m.get("role") == "user"]
        aggregated["total_messages"] += len(user_messages)

        # Calculate duration
        start_time = transcript.get("start_time")
        end_time = transcript.get("end_time")
        if start_time and end_time:
            try:
                start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                end = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
                duration_hours = (end - start).total_seconds() / 3600
                aggregated["total_duration_hours"] += duration_hours
            except (ValueError, TypeError):
                pass

        # Count tools and detect languages from assistant messages
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                # Count tool uses from formatted content
                for tool_match in re.findall(r'\[Tool: (\w+)\]', content):
                    aggregated["tool_counts"][tool_match] = aggregated["tool_counts"].get(tool_match, 0) + 1

                # Detect languages from file paths in content
                for ext, lang in LANGUAGE_MAP.items():
                    if ext in content:
                        aggregated["languages"][lang] = aggregated["languages"].get(lang, 0) + 1

        # Count git commits (look for commit-related patterns)
        for msg in messages:
            content = str(msg.get("content", ""))
            if "git commit" in content.lower() or "[Tool: Bash]" in content and "commit" in content.lower():
                aggregated["git_commits"] += 1
                break  # Count max 1 commit per session

    # Aggregate facet data
    for facet in facets:
        if not facet:
            continue

        # Collect timestamps for date range
        if facet.get("start_time"):
            all_dates.append(facet["start_time"])
        if facet.get("end_time"):
            all_dates.append(facet["end_time"])

        # Merge goal_categories
        merge_count_dicts(aggregated["goal_categories"], facet.get("goal_categories", {}))

        # Count outcomes
        outcome = facet.get("outcome", "unknown")
        aggregated["outcomes"][outcome] = aggregated["outcomes"].get(outcome, 0) + 1

        # Merge satisfaction counts
        merge_count_dicts(aggregated["satisfaction"], facet.get("user_satisfaction_counts", {}))

        # Count helpfulness
        helpfulness = facet.get("claude_helpfulness", "unknown")
        aggregated["helpfulness"][helpfulness] = aggregated["helpfulness"].get(helpfulness, 0) + 1

        # Count session types
        session_type = facet.get("session_type", "unknown")
        aggregated["session_types"][session_type] = aggregated["session_types"].get(session_type, 0) + 1

        # Merge friction counts
        merge_count_dicts(aggregated["friction"], facet.get("friction_counts", {}))

        # Count primary successes
        success = facet.get("primary_success", "none")
        aggregated["primary_successes"][success] = aggregated["primary_successes"].get(success, 0) + 1

        # Collect summaries (include outcome/helpfulness for prompt formatting)
        if facet.get("brief_summary"):
            aggregated["summaries"].append({
                "session_id": facet.get("session_id"),
                "summary": facet.get("brief_summary"),
                "underlying_goal": facet.get("underlying_goal", ""),
                "friction_detail": facet.get("friction_detail", ""),
                "outcome": facet.get("outcome", "unknown"),
                "helpfulness": facet.get("claude_helpfulness", "unknown"),
            })

    # Get top 8 for various categories
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

    aggregated["top_tools"] = sorted(
        aggregated["tool_counts"].items(),
        key=lambda x: x[1],
        reverse=True
    )[:8]

    # Calculate date range and days active
    if all_dates:
        sorted_dates = sorted(all_dates)
        start_date = sorted_dates[0][:10] if sorted_dates[0] else None
        end_date = sorted_dates[-1][:10] if sorted_dates[-1] else None
        aggregated["date_range"] = {"start": start_date, "end": end_date}

        # Calculate days active (unique dates)
        unique_dates = set()
        for d in all_dates:
            if d:
                unique_dates.add(d[:10])  # Just the date part
        aggregated["days_active"] = len(unique_dates)

    return aggregated




# ============================================================================
# Stage 3: Map-Reduce Analysis (7 Parallel Prompts)
# ============================================================================

ANALYSIS_PROMPTS = [
    "project_areas",
    "interaction_style",
    "what_works",
    "friction_analysis",
    "suggestions",
    "on_the_horizon",
    "fun_ending"
]


def format_prompt_data(aggregated: dict, facets: list[dict]) -> str:
    """Format data for analysis prompts matching real Claude CLI format."""
    # 1. Format session summaries as bullet points with outcome/helpfulness
    summaries = aggregated.get('summaries', [])
    session_summaries = "\n".join(
        f"- {s.get('summary', '')} ({s.get('outcome', 'unknown')}, {s.get('helpfulness', 'unknown')})"
        for s in summaries[:50]
    )

    # 2. Format friction details as bullet points (only non-empty)
    friction_details = "\n".join(
        f"- {s['friction_detail']}"
        for s in summaries[:20]
        if s.get('friction_detail')
    )

    # 3. Format user instructions (from facets if available)
    user_instructions = "\n".join(
        f"- {instr}"
        for facet in facets[:15]
        for instr in facet.get('user_instructions_to_claude', [])
    ) or "None captured"

    # Helper to filter out zero values from dicts
    def filter_zeros(d: dict) -> dict:
        return {k: v for k, v in d.items() if v}

    # Filter zeros from top lists (list of tuples)
    def filter_zero_tuples(lst: list) -> list:
        return [(k, v) for k, v in lst if v]

    # 4. Slim down aggregated data to match real CLI format (filter zeros like real CLI)
    slimmed = {
        "sessions": aggregated.get("total_sessions", 0),
        "analyzed": aggregated.get("sessions_with_facets", 0),
        "date_range": aggregated.get("date_range", {}),
        "messages": aggregated.get("total_messages", 0),
        "hours": round(aggregated.get("total_duration_hours", 0)) if aggregated.get("total_duration_hours") else 0,
        "commits": aggregated.get("git_commits", 0),
        "top_tools": filter_zero_tuples(aggregated.get("top_tools", []))[:8] if aggregated.get("top_tools") else list(filter_zeros(aggregated.get("tool_counts", {})).items())[:8],
        "top_goals": filter_zero_tuples(aggregated.get("top_goals", []))[:8],
        "outcomes": filter_zeros(aggregated.get("outcomes", {})),
        "satisfaction": filter_zeros(aggregated.get("satisfaction", {})),
        "friction": filter_zeros(aggregated.get("friction", {})),
        "success": filter_zeros(aggregated.get("primary_successes", {})),
        "languages": filter_zeros(aggregated.get("languages", {})),
    }

    return f"""{json.dumps(slimmed, indent=2)}

SESSION SUMMARIES:
{session_summaries}

FRICTION DETAILS:
{friction_details}

USER INSTRUCTIONS TO CLAUDE:
{user_instructions}"""


@weave.op(
    color="green",
    call_display_name=lambda call: f"analyze:{call.inputs['prompt_name']}"
)
def run_analysis_prompt(prompt_name: str, aggregated: dict, facets: list[dict], api_key: str | None = None) -> tuple[str, dict]:
    """Run a single analysis prompt and return (name, result)."""
    prompt_template = load_prompt(prompt_name)

    # Build context using real Claude CLI format
    context = format_prompt_data(aggregated, facets)

    full_prompt = prompt_template + "\n\nDATA:\n" + context

    result = call_llm(full_prompt, api_key=api_key)
    return prompt_name, result




# ============================================================================
# Stage 4: At-a-Glance Synthesis
# ============================================================================

@weave.op(color="cyan")
def generate_at_a_glance(analysis: dict, aggregated: dict, api_key: str | None = None) -> dict:
    """Generate at-a-glance synthesis from analysis results."""
    prompt_template = load_prompt("at_a_glance")

    # Format analysis sections as bullet points (like real CLI)
    project_areas = analysis.get("project_areas", {})
    areas_bullets = "\n".join(
        f"- {a.get('name', '')}: {a.get('description', '')}"
        for a in project_areas.get("areas", [])
    ) or ""

    what_works = analysis.get("what_works", {})
    wins_bullets = "\n".join(
        f"- {w.get('title', '')}: {w.get('description', '')}"
        for w in what_works.get("impressive_workflows", [])
    ) or ""

    friction = analysis.get("friction_analysis", {})
    friction_bullets = "\n".join(
        f"- {c.get('category', '')}: {c.get('description', '')}"
        for c in friction.get("categories", [])
    ) or ""

    suggestions = analysis.get("suggestions", {})
    features_bullets = "\n".join(
        f"- {f.get('feature', '')}: {f.get('one_liner', '')}"
        for f in suggestions.get("features_to_try", [])
    ) or ""

    patterns_bullets = "\n".join(
        f"- {p.get('title', '')}: {p.get('suggestion', '')}"
        for p in suggestions.get("usage_patterns", [])
    ) or ""

    horizon = analysis.get("on_the_horizon", {})
    horizon_bullets = "\n".join(
        f"- {o.get('title', '')}: {o.get('whats_possible', '')}"
        for o in horizon.get("opportunities", [])
    ) or ""

    # Replace template variables with formatted bullet points
    prompt = prompt_template.replace("${aggregated_data}", json.dumps(aggregated, indent=2))
    prompt = prompt.replace("${project_areas}", areas_bullets)
    prompt = prompt.replace("${what_works}", wins_bullets)
    prompt = prompt.replace("${friction_analysis}", friction_bullets)
    prompt = prompt.replace("${suggestions.features_to_try}", features_bullets)
    prompt = prompt.replace("${suggestions.usage_patterns}", patterns_bullets)
    prompt = prompt.replace("${on_the_horizon}", horizon_bullets)

    return call_llm(prompt, api_key=api_key)




# ============================================================================
# Full Pipeline (Core Logic)
# ============================================================================

from report_template import generate_html_report


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
) -> dict:
    """
    Run the full insights pipeline on transcripts.

    Args:
        transcripts: List of parsed session transcripts
        api_key: W&B API key (or set WANDB_API_KEY env var)
        workers: Number of parallel workers
        log: Optional callback for log messages

    Returns:
        HTML report bytes
    """
    def emit(msg: str):
        if log:
            log(msg)

    # Filter valid transcripts
    valid_transcripts = [t for t in transcripts if not should_skip_session(t)]
    skipped = len(transcripts) - len(valid_transcripts)
    emit(f"Processing {len(valid_transcripts)} sessions ({skipped} skipped) with {workers} workers...")

    # Stage 1: Extract facets in parallel
    emit("\n[Stage 1] Extracting facets...")

    def extract_worker(t):
        return extract_facets_from_transcript(t, api_key)

    facets, errors = parallel_llm_map(
        items=valid_transcripts,
        worker_fn=extract_worker,
        max_workers=workers,
        on_progress=lambda c, t: emit(f"  Facets: {c}/{t}")
    )
    emit(f"  Extracted {len(facets)} facets ({len(errors)} errors)")

    # Stage 2: Aggregate (pure Python) - pass transcripts for session-level stats
    emit("\n[Stage 2] Aggregating facets...")
    aggregated = aggregate_facets_impl(facets, valid_transcripts)

    # Stage 3: Run analysis prompts in parallel
    emit("\n[Stage 3] Running 7 analysis prompts...")

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
    emit("\n[Stage 4] Generating synthesis...")
    at_a_glance = generate_at_a_glance(analysis, aggregated, api_key)

    # Calculate date range from transcripts
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

    # Generate HTML report and set as call view
    emit("\nGenerating HTML report...")
    html = generate_html_report(report)
    emit(f"Pipeline complete! Report size: {len(html) / 1024:.1f} KB")

    # Set the report as a view on the call page
    weave.set_view("report", Content.from_text(html, extension=".html"))

    return {
        "session_count": len(transcripts),
        "analyzed_count": len(facets),
        "date_range": date_range,
        "generated_at": report["generated_at"],
        "html": html,
    }


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


# ============================================================================
# Weave Initialization (optional)
# ============================================================================

_weave_initialized = False
_weave_url = None


def init_weave(project: str = "claude-code-insights") -> str:
    """
    Initialize Weave tracing (optional).

    Call this before running the pipeline if you want tracing.

    Returns:
        The Weave project URL
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
