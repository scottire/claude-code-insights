# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.10.0",
#     "weave",
#     "openai",
#     "claude-code-insights @ git+https://github.com/scottire/claude-code-insights.git",
# ]
# ///
"""
Claude Code Insights Pipeline

Analyze your Claude Code sessions to see what's working, where things go wrong,
and get suggestions for improvement.

Run with:
    uvx marimo edit --sandbox insights_sandbox.py
"""

import marimo

__generated_with = "0.19.8"
app = marimo.App(width="full")


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
                            "project": _project_dir.name[:50] + "..." if len(_project_dir.name) > 50 else _project_dir.name,
                            "modified": datetime.fromtimestamp(_stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                            "size_kb": round(_stat.st_size / 1024, 1),
                        })

    # Sort by modified time (newest first)
    _session_files.sort(key=lambda x: x["modified"], reverse=True)
    session_files = _session_files
    return (session_files,)


@app.cell
def _(mo, session_files):
    # Create table data for selection
    _table_data = [
        {
            "project": s["project"],
            "modified": s["modified"],
            "size_kb": s["size_kb"],
        }
        for s in session_files[:100]  # Limit to 100 most recent
    ]

    session_table = mo.ui.table(
        data=_table_data,
        selection="multi",
        label=f"Select sessions ({len(session_files)} available)",
    ) if session_files else None
    return (session_table,)


@app.cell
def _(session_table):
    selected_files = [row['project'] for row in session_table.value]
    return (selected_files,)


@app.cell
def _(mo, selected_files):
    # Load button
    load_button = mo.ui.run_button(
        label=f"Load {len(selected_files)} Sessions" if selected_files else "Select sessions first",
        disabled=not selected_files,
    )
    return (load_button,)


@app.cell
def _(load_button, selected_files):
    # Load the selected sessions
    transcripts = []

    if load_button.value and selected_files:
        from insights.pipeline import parse_jsonl_session
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
    # Run button
    run_button = mo.ui.run_button(
        label="Run Pipeline",
        disabled=not transcripts,
    )
    return (run_button,)


@app.cell
def _(
    api_key_input,
    claude_dir,
    load_button,
    mo,
    run_button,
    session_files,
    transcripts,
):
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
        load_button,
        mo.md(f"*{len(transcripts)} loaded*") if transcripts else None,
        run_button if transcripts else None,
    ])

    mo.sidebar(_sidebar)
    return


@app.cell
def _(mo):
    mo.md("""
    # Claude Code Insights

    Analyze your Claude Code sessions to see what's working, where things go wrong, and get suggestions.

    **Select sessions from the table below**, then click **Load** and **Run Pipeline** in the sidebar.
    """)
    return


@app.cell
def _(mo, session_table):
    # Show session table
    if session_table:
        mo.vstack([
            mo.md("## Select Sessions"),
            session_table,
        ])
    else:
        mo.callout("No sessions found in ~/.claude/projects/", kind="warn")
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## How It Works
    """)
    return


@app.cell
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


@app.cell
def _(mo):
    mo.accordion({
        "Stage 1: Facet Extraction": mo.vstack([
            mo.md("""
    Each session gets analyzed by an LLM to extract structured data:

    - **Goal**: What were you trying to do?
    - **Outcome**: Did it work?
    - **Friction**: What went wrong?

    The prompt tells the LLM to only count what *you* asked for, not what Claude decided to do on its own:
            """),
            mo.md("""
    ```
    goal_categories: Count ONLY what the USER explicitly asked for.
      - DO NOT count Claude's autonomous codebase exploration
      - ONLY count when user says "can you...", "please...", "I need..."

    friction_counts: Be specific about what went wrong.
      - misunderstood_request: Claude interpreted incorrectly
      - wrong_approach: Right goal, wrong solution
      - buggy_code: Code didn't work
    ```
            """),
        ]),

        "Stage 2: Aggregation": mo.md("""
    Pure Python, no LLM. Combines facets into totals:

    - Merge goal counts across sessions
    - Sum friction types
    - Calculate outcome distributions
    - Collect summaries for the next stage
        """),

        "Stage 3: Parallel Analysis": mo.vstack([
            mo.md("Seven prompts run in parallel, each looking at your data differently:"),
            mo.ui.table(
                data=[
                    {"Prompt": "project_areas", "What it finds": "Projects and areas you work on"},
                    {"Prompt": "interaction_style", "What it finds": "How you work with Claude"},
                    {"Prompt": "what_works", "What it finds": "Your best workflows"},
                    {"Prompt": "friction_analysis", "What it finds": "Recurring problems"},
                    {"Prompt": "suggestions", "What it finds": "Features to try"},
                    {"Prompt": "on_the_horizon", "What it finds": "Future possibilities"},
                    {"Prompt": "fun_ending", "What it finds": "A memorable moment"},
                ],
                selection=None,
            ),
        ]),

        "Stage 4: Synthesis": mo.md("""
    Combines all seven analyses into a summary:

    1. **What's working** - Your style and wins
    2. **What's hindering** - Problems on both sides
    3. **Quick wins** - Easy things to try
    4. **Ambitious workflows** - What's coming with better models
        """),
    })
    return


@app.cell
def _(api_key_input, mo, run_button, transcripts):
    mo.stop(not run_button.value)

    if not api_key_input.value:
        mo.stop(True, mo.callout("Enter your W&B API key in the sidebar.", kind="warn"))

    if not transcripts:
        mo.stop(True, mo.callout("Load some sessions first.", kind="warn"))

    from insights.pipeline import init_weave, run_insights_pipeline

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

    html_bytes = run_insights_pipeline(
        transcripts=transcripts,
        api_key=api_key_input.value,
        workers=7,
        log=_log,
    )

    html_report = html_bytes.decode("utf-8")
    pipeline_logs = "\n".join(_logs)

    mo.vstack([
        mo.callout(mo.md(f"Done! [View traces]({weave_url})"), kind="success"),
        mo.accordion({"Pipeline logs": mo.md(f"```\n{pipeline_logs}\n```")}),
    ])
    return html_bytes, html_report


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Report
    """)
    return


@app.cell
def _(html_bytes, html_report, mo, run_button):
    mo.stop(not run_button.value)
    mo.stop(not html_report)

    download_button = mo.download(
        data=html_bytes,
        filename="claude_code_insights.html",
        mimetype="text/html",
        label="Download Report",
    )

    mo.hstack([download_button, mo.md(f"*{len(html_bytes) / 1024:.0f} KB*")])
    return


@app.cell
def _(html_report, mo, run_button):
    mo.stop(not run_button.value)
    mo.stop(not html_report)

    mo.Html(html_report)
    return


if __name__ == "__main__":
    app.run()
