"""
Claude Code Insights Pipeline

Analyze your Claude Code sessions to see what's working, where things go wrong,
and get suggestions for improvement.

Run with:
    uv run marimo edit notebook.py
"""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # What does Claude think about your coding sessions?

    The `/insights` command in Claude Code reads your session transcripts and generates a report about how you work. It tells you where things go well, where they don't, and what you might try differently.

    I reverse-engineered the pipeline to see how it works. Turns out it's a map-reduce pattern: extract structured data from each session, aggregate it, run parallel analysis prompts, then synthesize into a report.

    This notebook runs that same pipeline. You can step through each stage to see what's happening, or just hit run and get your report.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **To run the pipeline:** Get a [W&B API key](https://wandb.ai/authorize), paste it in the sidebar, select some sessions below, and click Run.

    **To explore how it works:** Expand the stages below, then step through each one after loading sessions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(
        """
        flowchart LR
            A[Sessions] --> B[1. Extract Facets]
            B --> C[2. Aggregate]
            C --> D[3. Analyze x7]
            D --> E[4. Synthesize]
            E --> F[Report]

            style A fill:#e3f2fd
            style F fill:#e8f5e9
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Stage 1: Facet extraction": mo.md("""
    An LLM reads each session transcript and extracts structured data: what you were trying to do, whether it worked, what went wrong. It outputs JSON with fixed categories like `debug_investigate`, `implement_feature`, `misunderstood_request`, etc.

    The prompt is strict about only counting what's explicit in the transcript - no guessing intent.
        """),

        "Stage 2: Aggregation": mo.md("""
    Pure Python. Merges facet counts across sessions, calculates distributions, collects the brief summaries for the next stage. No LLM here.
        """),

        "Stage 3: Parallel analysis": mo.vstack([
            mo.md("Seven prompts run in parallel, each looking at the aggregated data differently:"),
            mo.ui.table(
                data=[
                    {"Prompt": "project_areas", "Looks for": "What projects you work on"},
                    {"Prompt": "interaction_style", "Looks for": "How you talk to Claude"},
                    {"Prompt": "what_works", "Looks for": "Your best workflows"},
                    {"Prompt": "friction_analysis", "Looks for": "Recurring problems"},
                    {"Prompt": "suggestions", "Looks for": "Features you might try"},
                    {"Prompt": "on_the_horizon", "Looks for": "What better models could unlock"},
                    {"Prompt": "fun_ending", "Looks for": "A memorable moment"},
                ],
                selection=None,
            ),
        ]),

        "Stage 4: Synthesis": mo.md("""
    Takes the seven analysis outputs and condenses them into four sections for the report: what's working, what's not, quick wins, and future possibilities.
        """),
    })
    return


@app.cell
def _(mo, projects_data, session_files):
    project_table = mo.ui.table(
        data=projects_data,
        selection="multi",
        label=f"Select projects ({len(projects_data)} projects, {len(session_files)} sessions)",
    ) if projects_data else None
    project_table
    return (project_table,)


@app.cell
def _(project_table, session_files):
    # Get all session files for selected projects
    selected_projects = {row['project'] for row in project_table.value} if project_table.value else set()
    selected_files = [s["file"] for s in session_files if s["project"] in selected_projects]
    return (selected_files,)


@app.cell
def _(mo, selected_files):
    load_button = mo.ui.run_button(
        label=f"Load {len(selected_files)} sessions" if selected_files else "Select projects first",
        disabled=not selected_files,
    )
    return (load_button,)


@app.cell
def _(load_button, selected_files):
    transcripts = []
    if load_button.value and selected_files:
        from pipeline import parse_jsonl_session
        for _f in selected_files:
            try:
                _t = parse_jsonl_session(_f)
                if _t["messages"]:
                    transcripts.append(_t)
            except Exception:
                pass
    return (transcripts,)


@app.cell
def _(mo, transcripts):
    run_button = mo.ui.run_button(
        label=f"Run pipeline ({len(transcripts)} sessions)" if transcripts else "Load sessions first",
        disabled=not transcripts,
    )
    return (run_button,)


@app.cell
def _(load_button, mo, run_button):
    mo.hstack([load_button, run_button], justify="start", gap=1)
    return


@app.cell(hide_code=True)
def _(mo, transcripts):
    mo.stop(not transcripts)
    mo.md("""
    ---
    ## Step through: facet extraction

    Before running the full pipeline, try extracting facets from a single session. The LLM reads the transcript and outputs structured JSON.
    """)
    return


@app.cell
def _(mo, transcripts):
    # Button to extract facets from first session
    extract_button = mo.ui.run_button(
        label="Extract Facets from First Session",
        disabled=not transcripts,
    )
    extract_button
    return (extract_button,)


@app.cell
def _(api_key_input, extract_button, mo, transcripts):
    mo.stop(not extract_button.value)
    mo.stop(not transcripts)

    if not api_key_input.value:
        mo.stop(True, mo.callout("Enter your W&B API key in the sidebar first. This uses a model hosted on W&B to power the pipeline.", kind="warn"))

    from pipeline import format_transcript_compact, should_skip_session

    # Find first transcript that won't be skipped
    first_transcript = None
    for t in transcripts:
        if not should_skip_session(t):
            first_transcript = t
            break

    if not first_transcript:
        mo.stop(True, mo.callout("All loaded sessions are too short (need 2+ user messages and 1+ minute duration). Try loading different sessions.", kind="warn"))

    transcript_preview = format_transcript_compact(first_transcript)
    user_msg_count = len([m for m in first_transcript.get("messages", []) if m.get("role") == "user"])

    mo.vstack([
        mo.md(f"**Session:** `{first_transcript.get('session_id', 'unknown')[:20]}...` ({user_msg_count} user messages)"),
        mo.accordion({
            "Transcript (compact format sent to LLM)": mo.md(f"```\n{transcript_preview[:3000]}{'...' if len(transcript_preview) > 3000 else ''}\n```")
        })
    ])
    return (first_transcript,)


@app.cell
def _(api_key_input, extract_button, first_transcript, mo):
    mo.stop(not extract_button.value)

    from pipeline import extract_facets_from_transcript

    with mo.status.spinner("Extracting facets..."):
        facet_result = extract_facets_from_transcript(first_transcript, api_key=api_key_input.value)
    return (facet_result,)


@app.cell(hide_code=True)
def _(extract_button, facet_result, mo):
    mo.stop(not extract_button.value)
    mo.stop(not facet_result)
    mo.vstack([
        mo.md("### Extracted Facets"),
        mo.ui.table(
            data=[
                {"Field": "Goal", "Value": facet_result.get("underlying_goal", "N/A")},
                {"Field": "Outcome", "Value": facet_result.get("outcome", "N/A")},
                {"Field": "Session Type", "Value": facet_result.get("session_type", "N/A")},
                {"Field": "Claude Helpfulness", "Value": facet_result.get("claude_helpfulness", "N/A")},
                {"Field": "Primary Success", "Value": facet_result.get("primary_success", "N/A")},
                {"Field": "Summary", "Value": facet_result.get("brief_summary", "N/A")},
            ],
            selection=None,
        ),
        mo.accordion({
            "Goal Categories": mo.json(facet_result.get("goal_categories", {})),
            "User Satisfaction": mo.json(facet_result.get("user_satisfaction_counts", {})),
            "Friction": mo.json(facet_result.get("friction_counts", {})) if facet_result.get("friction_counts") else mo.md("*No friction detected*"),
            "Friction Detail": mo.md(facet_result.get("friction_detail", "*None*") or "*None*"),
            "User Instructions": mo.md("\n".join(f"- {i}" for i in facet_result.get("user_instructions_to_claude", [])) or "*None captured*"),
            "Full JSON": mo.json(facet_result),
        }),
    ])
    return


@app.cell(hide_code=True)
def _(extract_button, mo):
    mo.stop(not extract_button.value)
    mo.md("""
    ---
    ## Step through: aggregation

    Combines facets into totals, merging counts and collecting summaries. No LLM here.
    """)
    return


@app.cell
def _(extract_button, facet_result, mo):
    aggregate_button = mo.ui.run_button(
        label="Run Aggregation",
        disabled=not (extract_button.value and facet_result),
    )
    aggregate_button
    return (aggregate_button,)


@app.cell
def _(aggregate_button, facet_result, first_transcript, mo):
    mo.stop(not aggregate_button.value)

    from pipeline import aggregate_facets_impl

    # Run aggregation on our single facet (in real pipeline this would be many)
    aggregated = aggregate_facets_impl([facet_result], [first_transcript])

    mo.vstack([
        mo.md("### Input: Facets"),
        mo.md(f"*{1} facet from {1} session*"),
        mo.md("### Output: Aggregated Statistics"),
        mo.hstack([
            mo.stat(value=aggregated.get("total_sessions", 0), label="Sessions"),
            mo.stat(value=aggregated.get("total_messages", 0), label="Messages"),
            mo.stat(value=f"{aggregated.get('total_duration_hours', 0):.1f}h", label="Duration"),
        ]),
        mo.accordion({
            "Top Goals": mo.json(aggregated.get("top_goals", [])),
            "Outcomes": mo.json(aggregated.get("outcomes", {})),
            "Satisfaction": mo.json(aggregated.get("satisfaction", {})),
            "Top Friction": mo.json(aggregated.get("top_friction", [])),
            "Summaries (for next stage)": mo.json(aggregated.get("summaries", [])),
            "Full Aggregated Data": mo.json(aggregated),
        }),
    ])
    return (aggregated,)


@app.cell(hide_code=True)
def _(aggregate_button, mo):
    mo.stop(not aggregate_button.value)
    mo.md("""
    ---
    ## Step through: analysis prompts

    In the full pipeline, seven prompts run in parallel. Pick one to try:
    """)
    return


@app.cell
def _(aggregate_button, mo):
    prompt_dropdown = mo.ui.dropdown(
        options=[
            "project_areas",
            "interaction_style",
            "what_works",
            "friction_analysis",
            "suggestions",
            "on_the_horizon",
            "fun_ending",
        ],
        value="what_works",
        label="Analysis Prompt",
    )
    analyze_button = mo.ui.run_button(
        label="Run Analysis",
        disabled=not aggregate_button.value,
    )
    mo.hstack([prompt_dropdown, analyze_button])
    return analyze_button, prompt_dropdown


@app.cell
def _(aggregated, analyze_button, facet_result, mo, prompt_dropdown):
    mo.stop(not analyze_button.value)

    from pipeline import format_prompt_data

    # Show the input that goes to the LLM
    prompt_data = format_prompt_data(aggregated, [facet_result])

    mo.vstack([
        mo.md(f"### Input to `{prompt_dropdown.value}` prompt"),
        mo.accordion({
            "Formatted data sent to LLM": mo.md(f"```\n{prompt_data[:2000]}{'...' if len(prompt_data) > 2000 else ''}\n```"),
        }),
    ])
    return


@app.cell
def _(
    aggregated,
    analyze_button,
    api_key_input,
    facet_result,
    mo,
    prompt_dropdown,
):
    mo.stop(not analyze_button.value)

    from pipeline import run_analysis_prompt

    with mo.status.spinner(f"Running {prompt_dropdown.value} analysis..."):
        prompt_name, analysis_result = run_analysis_prompt(
            prompt_dropdown.value,
            aggregated,
            [facet_result],
            api_key=api_key_input.value
        )

    mo.vstack([
        mo.md(f"### Output from `{prompt_name}`"),
        mo.json(analysis_result),
    ])
    return (analysis_result,)


@app.cell(hide_code=True)
def _(analyze_button, mo):
    mo.stop(not analyze_button.value)
    mo.md("""
    ---
    ## Step through: synthesis

    Takes all seven analysis outputs and condenses them into four report sections. (Here we only have one analysis, so the output will be sparse.)
    """)
    return


@app.cell
def _(analyze_button, mo):
    synthesize_button = mo.ui.run_button(
        label="Run Synthesis",
        disabled=not analyze_button.value,
    )
    synthesize_button
    return (synthesize_button,)


@app.cell
def _(aggregated, analysis_result, mo, prompt_dropdown, synthesize_button):
    mo.stop(not synthesize_button.value)

    # Build a partial analysis dict with what we have
    partial_analysis = {prompt_dropdown.value: analysis_result}

    mo.vstack([
        mo.md("### Input to Synthesis"),
        mo.accordion({
            "Analysis Results": mo.json(partial_analysis),
            "Aggregated Stats": mo.json({
                "sessions": aggregated.get("total_sessions"),
                "messages": aggregated.get("total_messages"),
                "outcomes": aggregated.get("outcomes"),
            }),
        }),
    ])
    return (partial_analysis,)


@app.cell
def _(aggregated, api_key_input, mo, partial_analysis, synthesize_button):
    mo.stop(not synthesize_button.value)

    from pipeline import generate_at_a_glance

    with mo.status.spinner("Generating at-a-glance synthesis..."):
        at_a_glance = generate_at_a_glance(partial_analysis, aggregated, api_key=api_key_input.value)

    mo.vstack([
        mo.md("### Output: At-a-Glance Summary"),
        mo.callout(mo.md("This is the final synthesis that appears at the top of the report!"), kind="success"),
        mo.accordion({
            "What's Working": mo.md(at_a_glance.get("whats_working", "*Not generated*")),
            "What's Hindering": mo.md(at_a_glance.get("whats_hindering", "*Not generated*")),
            "Quick Wins": mo.md(at_a_glance.get("quick_wins", "*Not generated*")),
            "Ambitious Workflows": mo.md(at_a_glance.get("ambitious_workflows", "*Not generated*")),
            "Full JSON": mo.json(at_a_glance),
        }),
    ])
    return


@app.cell
def _(api_key_input, claude_dir, mo, session_files):
    # Sidebar content
    _sidebar = mo.vstack([
        mo.md("## Configuration"),
        api_key_input,
        mo.md("[Get API key](https://wandb.ai/authorize)"),
        mo.md("---"),
        mo.md("## Sessions"),
        mo.md(f"`{claude_dir}`"),
        mo.md(f"**{len(session_files)}** sessions found") if session_files else mo.callout("No sessions found", kind="warn"),
        mo.md("---"),
    ])

    mo.sidebar(_sidebar)
    return


@app.cell
def _(api_key_input, mo, run_button, transcripts):
    mo.stop(not run_button.value)

    if not api_key_input.value:
        mo.stop(True, mo.callout("Enter your W&B API key in the sidebar.", kind="warn"))

    if not transcripts:
        mo.stop(True, mo.callout("Load some sessions first.", kind="warn"))

    from pipeline import init_weave, run_insights_pipeline

    weave_url = init_weave("claude-code-insights")

    mo.md(f"**Traces:** [{weave_url}]({weave_url})")
    return run_insights_pipeline, weave_url


@app.cell
def _(
    api_key_input,
    mo,
    run_button,
    run_insights_pipeline,
    transcripts,
    weave_url,
):
    mo.stop(not run_button.value)

    _logs = []

    def _log(msg):
        _logs.append(msg)

    pipeline_result = run_insights_pipeline(
        transcripts=transcripts,
        api_key=api_key_input.value,
        workers=7,
        log=_log,
    )

    html_report = pipeline_result["html"]
    pipeline_logs = "\n".join(_logs)

    mo.vstack([
        mo.callout(mo.md(f"Done! Analyzed {pipeline_result['analyzed_count']} sessions. [View traces]({weave_url})"), kind="success"),
        mo.accordion({"Pipeline logs": mo.md(f"```\n{pipeline_logs}\n```")}),
    ])
    return (html_report,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Report
    """)
    return


@app.cell
def _(html_report, mo, run_button):
    mo.stop(not run_button.value)
    mo.stop(not html_report)

    html_bytes = html_report.encode("utf-8")
    download_button = mo.download(
        data=html_bytes,
        filename="claude_code_insights.html",
        mimetype="text/html",
        label="Download Report",
    )

    mo.hstack([download_button, mo.md(f"*{len(html_bytes) / 1024:.0f} KB*")])
    return


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    from datetime import datetime

    return Path, datetime, mo


@app.cell
def _(mo):
    # API key input
    api_key_input = mo.ui.text(
        label="W&B API Key",
        kind="password",
        placeholder="Paste your API key here",
        full_width=True,
    )
    return (api_key_input,)


@app.cell
def _(Path):
    # Check for Claude sessions
    claude_dir = Path.home() / ".claude" / "projects"
    claude_dir_exists = claude_dir.exists()
    return claude_dir, claude_dir_exists


@app.cell
def _(claude_dir, claude_dir_exists, datetime):
    def _decode_project_path(encoded_name: str) -> str:
        """Decode Claude's encoded project path (e.g. '-Users-scott-Dev-project' -> 'Dev/project')"""
        # Replace leading dash and convert dashes to slashes
        if encoded_name.startswith("-"):
            encoded_name = encoded_name[1:]
        decoded = encoded_name.replace("-", "/")
        # Get last 2 path components for readability
        parts = decoded.split("/")
        if len(parts) >= 2:
            return "/".join(parts[-2:])
        return parts[-1] if parts else encoded_name

    # Scan for sessions without loading them
    _session_files = []
    if claude_dir_exists:
        for _project_dir in claude_dir.glob("*"):
            if _project_dir.is_dir():
                for _f in _project_dir.glob("*.jsonl"):
                    if "subagents" not in str(_f):
                        _stat = _f.stat()
                        _session_files.append({
                            "file": _f,
                            "project_encoded": _project_dir.name,  # Keep original for matching
                            "project": _decode_project_path(_project_dir.name),
                            "modified": datetime.fromtimestamp(_stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                            "size_kb": round(_stat.st_size / 1024, 1),
                        })

    # Sort by modified time (newest first)
    _session_files.sort(key=lambda x: x["modified"], reverse=True)
    session_files = _session_files
    return (session_files,)


@app.cell
def _(session_files):
    # Group sessions by project
    from collections import Counter
    _project_counts = Counter(s["project"] for s in session_files)
    projects_data = [
        {"project": proj, "sessions": count}
        for proj, count in _project_counts.most_common()
    ]
    return (projects_data,)


if __name__ == "__main__":
    app.run()
