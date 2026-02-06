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

__generated_with = "0.10.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    from datetime import datetime
    return mo, Path, datetime


@app.cell
def _(mo):
    mo.md(
        """
        # Claude Code Insights

        This notebook analyzes your Claude Code sessions and generates a report showing:
        - What you've been working on
        - Where Claude helped (and where it got in the way)
        - Features worth trying

        The analysis runs through four stages, each building on the last.
        """
    )
    return


@app.cell
def _(mo):
    # API key input
    api_key_input = mo.ui.text(
        label="W&B API Key",
        kind="password",
        placeholder="Paste your API key here",
        full_width=True,
    )
    mo.vstack([
        api_key_input,
        mo.md("[Get your API key](https://wandb.ai/authorize)"),
    ])
    return (api_key_input,)


@app.cell
def _(Path):
    # Check for Claude sessions
    claude_dir = Path.home() / ".claude" / "projects"
    claude_dir_exists = claude_dir.exists()
    return claude_dir, claude_dir_exists


@app.cell
def _(claude_dir, claude_dir_exists, datetime, mo):
    # Scan for sessions without loading them
    session_files = []
    if claude_dir_exists:
        for project_dir in claude_dir.glob("*"):
            if project_dir.is_dir():
                for f in project_dir.glob("*.jsonl"):
                    if "subagents" not in str(f):
                        stat = f.stat()
                        session_files.append({
                            "file": f,
                            "project": project_dir.name[:40] + "..." if len(project_dir.name) > 40 else project_dir.name,
                            "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                            "size_kb": round(stat.st_size / 1024, 1),
                        })

    # Sort by modified time (newest first)
    session_files.sort(key=lambda x: x["modified"], reverse=True)

    if not claude_dir_exists:
        mo.callout(f"No Claude sessions found at `{claude_dir}`", kind="warn")

    return (session_files,)


@app.cell
def _(mo, session_files):
    mo.stop(not session_files)

    # Create table data for selection
    table_data = [
        {
            "project": s["project"],
            "modified": s["modified"],
            "size_kb": s["size_kb"],
        }
        for s in session_files[:100]  # Limit to 100 most recent
    ]

    session_table = mo.ui.table(
        data=table_data,
        selection="multi",
        label=f"Select sessions to analyze ({len(session_files)} available)",
    )

    mo.vstack([
        mo.md("## 1. Select Sessions"),
        mo.md(f"Found **{len(session_files)} sessions** in `~/.claude/projects/`"),
        session_table,
        mo.md("*Select rows to include in analysis. Newest sessions shown first.*"),
    ])

    return session_table, table_data


@app.cell
def _(mo, session_files, session_table):
    mo.stop(not session_table.value)

    # Get selected session files
    selected_indices = [row["index"] for row in session_table.value] if session_table.value else []
    selected_files = [session_files[i]["file"] for i in selected_indices]

    mo.md(f"**{len(selected_files)} sessions selected**")

    return selected_files, selected_indices


@app.cell
def _(mo, selected_files):
    mo.stop(not selected_files)

    # Load button
    load_button = mo.ui.run_button(label=f"Load {len(selected_files)} Sessions")
    load_button

    return (load_button,)


@app.cell
def _(load_button, mo, selected_files):
    mo.stop(not load_button.value)

    from insights.pipeline import parse_jsonl_session

    # Load the selected sessions
    transcripts = []
    for f in selected_files:
        try:
            t = parse_jsonl_session(f)
            if t["messages"]:
                transcripts.append(t)
        except Exception:
            pass

    mo.callout(f"Loaded {len(transcripts)} sessions with messages", kind="success")

    return parse_jsonl_session, transcripts


@app.cell
def _(mo):
    mo.md(
        """
        ---
        ## 2. How the Pipeline Works

        The analysis happens in four stages. Each stage transforms the data for the next.
        """
    )
    return


@app.cell
def _(mo):
    mo.mermaid(
        """
        flowchart LR
            A[Your Sessions] --> B[Stage 1: Extract Facets]
            B --> C[Stage 2: Aggregate]
            C --> D[Stage 3: Analyze]
            D --> E[Stage 4: Synthesize]
            E --> F[Report]

            style A fill:#e3f2fd
            style F fill:#e8f5e9
        """
    )
    return


@app.cell
def _(mo):
    # Show what each stage does
    mo.accordion({
        "Stage 1: Facet Extraction (LLM)": mo.vstack([
            mo.md("""
Each session gets analyzed individually. The LLM extracts structured "facets":

- **Goal**: What were you trying to do?
- **Outcome**: Did it work? (fully/mostly/partially/not achieved)
- **Satisfaction**: How did you seem to feel about it?
- **Friction**: What went wrong? (misunderstood request, buggy code, wrong approach, etc.)

Here's a snippet of the prompt:
            """),
            mo.md("""
```
Analyze this Claude Code session and extract structured facets.

CRITICAL GUIDELINES:

1. goal_categories: Count ONLY what the USER explicitly asked for.
   - DO NOT count Claude's autonomous codebase exploration
   - ONLY count when user says "can you...", "please...", "I need..."

2. user_satisfaction_counts: Base ONLY on explicit user signals.
   - "Yay!", "great!", "perfect!" → happy
   - "thanks", "looks good" → satisfied
   - "that's not right", "try again" → dissatisfied

3. friction_counts: Be specific about what went wrong.
   - misunderstood_request: Claude interpreted incorrectly
   - wrong_approach: Right goal, wrong solution method
   - buggy_code: Code didn't work correctly
```
            """),
        ]),

        "Stage 2: Aggregation (No LLM)": mo.md("""
This stage is pure Python - no LLM needed. It combines all the individual facets into totals:

- Merge goal category counts across sessions
- Sum up friction types
- Calculate outcome distributions
- Collect session summaries for the next stage

Fast and deterministic. Just counting and merging dictionaries.
        """),

        "Stage 3: Map-Reduce Analysis (7 Parallel LLMs)": mo.vstack([
            mo.md("""
Seven prompts run in parallel, each analyzing your aggregated data from a different angle:
            """),
            mo.ui.table(
                data=[
                    {"Prompt": "project_areas", "Purpose": "What projects/areas you work on"},
                    {"Prompt": "interaction_style", "Purpose": "How you interact with Claude"},
                    {"Prompt": "what_works", "Purpose": "Your most impressive workflows"},
                    {"Prompt": "friction_analysis", "Purpose": "Where things consistently go wrong"},
                    {"Prompt": "suggestions", "Purpose": "Features and CLAUDE.md additions to try"},
                    {"Prompt": "on_the_horizon", "Purpose": "Ambitious workflows for future models"},
                    {"Prompt": "fun_ending", "Purpose": "A memorable moment from your sessions"},
                ],
                selection=None,
            ),
            mo.md("Running these in parallel makes it ~7x faster than running them one at a time."),
        ]),

        "Stage 4: Synthesis (LLM)": mo.md("""
The final stage combines all seven analyses into a 4-part "At a Glance" summary:

1. **What's working** - Your style and wins
2. **What's hindering** - Claude's fault vs. user-side friction
3. **Quick wins** - Features worth trying
4. **Ambitious workflows** - What becomes possible with better models

This runs last because it needs all the previous analyses as input.
        """),
    })
    return


@app.cell
def _(mo):
    mo.md("---\n## 3. Run the Pipeline")
    return


@app.cell
def _(mo):
    workers_input = mo.ui.slider(
        label="Parallel workers",
        start=1,
        stop=10,
        value=5,
        step=1,
    )
    workers_input
    return (workers_input,)


@app.cell
def _(mo, transcripts):
    mo.stop(not transcripts)
    run_button = mo.ui.run_button(label="Run Pipeline")
    run_button
    return (run_button,)


@app.cell
def _(api_key_input, mo, run_button, transcripts, workers_input):
    mo.stop(not run_button.value)

    if not api_key_input.value:
        mo.stop(True, mo.callout("Enter your W&B API key at the top of the page.", kind="warn"))

    if not transcripts:
        mo.stop(True, mo.callout("Load some sessions first.", kind="warn"))

    from insights.pipeline import init_weave, run_insights_pipeline

    # Initialize Weave
    weave_url = init_weave("claude-code-insights")

    mo.vstack([
        mo.md(f"**Traces:** [{weave_url}]({weave_url})"),
        mo.md("Running pipeline..."),
    ])

    return init_weave, run_insights_pipeline, weave_url


@app.cell
def _(api_key_input, mo, run_button, run_insights_pipeline, transcripts, weave_url, workers_input):
    mo.stop(not run_button.value)

    logs = []

    def log(msg):
        logs.append(msg)

    html_bytes = run_insights_pipeline(
        transcripts=transcripts,
        api_key=api_key_input.value,
        workers=workers_input.value,
        log=log,
    )

    html_report = html_bytes.decode("utf-8")

    mo.vstack([
        mo.callout(
            mo.md(f"Done! [View traces]({weave_url})"),
            kind="success"
        ),
        mo.accordion({
            "Pipeline logs": mo.md(f"```\n{chr(10).join(logs)}\n```"),
        }),
    ])

    return html_bytes, html_report, log, logs


@app.cell
def _(mo):
    mo.md("---\n## 4. Your Report")
    return


@app.cell
def _(html_bytes, html_report, mo, run_button):
    mo.stop(not run_button.value)
    mo.stop(not html_report)

    download = mo.download(
        data=html_bytes,
        filename="claude_code_insights.html",
        mimetype="text/html",
        label="Download Report",
    )

    mo.hstack([download, mo.md(f"*{len(html_bytes) / 1024:.0f} KB*")])

    return (download,)


@app.cell
def _(html_report, mo, run_button):
    mo.stop(not run_button.value)
    mo.stop(not html_report)

    mo.Html(html_report)
    return


if __name__ == "__main__":
    app.run()
