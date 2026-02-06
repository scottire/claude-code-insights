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
Claude Code Insights Pipeline - Marimo Notebook

Run with:
    uvx marimo edit --sandbox insights_sandbox.py

Or from GitHub:
    uvx marimo edit --sandbox https://raw.githubusercontent.com/USER/REPO/main/insights_sandbox.py
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # Claude Code Insights Pipeline

        This notebook analyzes your Claude Code session transcripts and generates
        an insights report showing:

        - **What you work on** - Project areas and task types
        - **How you use Claude** - Interaction patterns and style
        - **What's working** - Impressive workflows and wins
        - **Where things go wrong** - Friction points and issues
        - **Suggestions** - Features to try and workflow improvements

        ## Getting Started

        1. **Get your W&B API Key**: Go to [wandb.ai/settings](https://wandb.ai/settings) and copy your API key
        2. **Configure below**: Enter your API key and choose a transcript source
        3. **Run the pipeline**: Click "Run Pipeline" to generate your report

        ## Viewing Weave Traces

        After running, visit [wandb.ai](https://wandb.ai) → your project → "Weave" tab to see all traced LLM calls.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## Configuration")
    return


@app.cell
def _(mo):
    api_key_input = mo.ui.text(
        label="W&B API Key",
        kind="password",
        placeholder="Enter your Weights & Biases API key",
        full_width=True,
    )
    weave_project_input = mo.ui.text(
        label="Weave Project Name",
        value="claude-code-insights",
        full_width=True,
    )
    workers_input = mo.ui.slider(
        label="Parallel Workers",
        start=1,
        stop=10,
        value=5,
        step=1,
    )

    mo.vstack([
        api_key_input,
        weave_project_input,
        workers_input,
    ])
    return api_key_input, weave_project_input, workers_input


@app.cell
def _(mo):
    mo.md("## Transcript Source")
    return


@app.cell
def _(mo):
    source_tabs = mo.ui.tabs({
        "Local Sessions": "local",
        "Upload File": "upload",
        "Paste JSON": "paste",
    })
    source_tabs
    return (source_tabs,)


@app.cell
def _(mo, source_tabs):
    # Local sessions options
    local_limit = mo.ui.slider(
        label="Number of sessions to analyze",
        start=5,
        stop=50,
        value=10,
        step=5,
    )

    # File upload
    file_upload = mo.ui.file(
        label="Upload JSONL or JSON transcript file",
        filetypes=[".jsonl", ".json"],
        multiple=True,
    )

    # Paste JSON
    json_paste = mo.ui.text_area(
        label="Paste transcript JSON",
        placeholder='[{"session_id": "...", "messages": [...]}]',
        rows=10,
        full_width=True,
    )

    # Show appropriate UI based on selected tab
    if source_tabs.value == "local":
        source_ui = mo.vstack([
            mo.md("Analyze sessions from `~/.claude/projects/`"),
            local_limit,
        ])
    elif source_tabs.value == "upload":
        source_ui = file_upload
    else:
        source_ui = json_paste

    source_ui
    return file_upload, json_paste, local_limit, source_ui


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run Pipeline")
    run_button
    return (run_button,)


@app.cell
def _(
    api_key_input,
    file_upload,
    json_paste,
    local_limit,
    mo,
    run_button,
    source_tabs,
    weave_project_input,
    workers_input,
):
    import json
    from pathlib import Path

    mo.stop(not run_button.value, mo.md("*Click 'Run Pipeline' to start*"))

    # Validate API key
    if not api_key_input.value:
        mo.stop(True, mo.callout(
            "Please enter your W&B API Key above.",
            kind="warn"
        ))

    # Load transcripts based on source
    transcripts = []

    if source_tabs.value == "local":
        claude_dir = Path.home() / ".claude" / "projects"
        if not claude_dir.exists():
            mo.stop(True, mo.callout(
                f"Claude projects directory not found: {claude_dir}",
                kind="danger"
            ))

        from insights.pipeline import load_transcripts_from_claude_dir
        transcripts = load_transcripts_from_claude_dir(claude_dir, limit=local_limit.value)

    elif source_tabs.value == "upload":
        if not file_upload.value:
            mo.stop(True, mo.callout("Please upload a transcript file.", kind="warn"))

        from insights.pipeline import parse_jsonl_session

        for uploaded_file in file_upload.value:
            content = uploaded_file.contents.decode("utf-8")
            if uploaded_file.name.endswith(".jsonl"):
                # Write to temp file and parse
                import tempfile
                with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
                    f.write(content)
                    temp_path = Path(f.name)
                transcripts.append(parse_jsonl_session(temp_path))
                temp_path.unlink()
            else:
                data = json.loads(content)
                if isinstance(data, list):
                    transcripts.extend(data)
                else:
                    transcripts.append(data)

    else:  # paste
        if not json_paste.value.strip():
            mo.stop(True, mo.callout("Please paste transcript JSON.", kind="warn"))

        data = json.loads(json_paste.value)
        if isinstance(data, list):
            transcripts = data
        else:
            transcripts = [data]

    if not transcripts:
        mo.stop(True, mo.callout("No transcripts found to analyze.", kind="warn"))

    mo.md(f"**Loaded {len(transcripts)} transcripts**")
    return (json, Path, transcripts,)


@app.cell
def _(
    api_key_input,
    mo,
    run_button,
    transcripts,
    weave_project_input,
    workers_input,
):
    mo.stop(not run_button.value)
    mo.stop(not transcripts)

    from insights.pipeline import init_weave, run_insights_pipeline

    # Initialize Weave
    init_weave(weave_project_input.value)

    # Track progress
    progress_status = mo.status.spinner("Running pipeline...")

    # Run the pipeline
    with progress_status:
        html_bytes = run_insights_pipeline(
            transcripts=transcripts,
            api_key=api_key_input.value,
            workers=workers_input.value,
        )

    html_report = html_bytes.decode("utf-8")
    return (html_bytes, html_report, init_weave, progress_status, run_insights_pipeline,)


@app.cell
def _(html_bytes, html_report, mo, run_button):
    mo.stop(not run_button.value)
    mo.stop(not html_report)

    mo.md("## Results")
    return


@app.cell
def _(html_bytes, html_report, mo, run_button):
    mo.stop(not run_button.value)
    mo.stop(not html_report)

    # Download button
    download = mo.download(
        data=html_bytes,
        filename="claude_code_insights.html",
        mimetype="text/html",
        label="Download HTML Report",
    )

    mo.hstack([
        download,
        mo.md(f"*Report size: {len(html_bytes) / 1024:.1f} KB*"),
    ])
    return (download,)


@app.cell
def _(html_report, mo, run_button):
    mo.stop(not run_button.value)
    mo.stop(not html_report)

    # Show the HTML report inline
    mo.Html(html_report)
    return


@app.cell
def _(mo):
    mo.md(
        """
        ---

        ## About This Pipeline

        The Claude Code Insights pipeline runs these stages:

        1. **Facet Extraction** - Analyzes each session transcript to extract structured data about goals, outcomes, friction, and satisfaction
        2. **Aggregation** - Combines facets into statistics (pure Python, no LLM)
        3. **Analysis** - Runs 7 parallel LLM prompts for deep analysis:
           - Project areas
           - Interaction style
           - What works
           - Friction analysis
           - Suggestions
           - On the horizon
           - Fun ending
        4. **Synthesis** - Generates the "At a Glance" summary

        All LLM calls are traced via [Weave](https://wandb.ai/site/weave) so you can inspect inputs, outputs, and latencies.

        ### Finding Your Claude Code Sessions

        Sessions are stored in `~/.claude/projects/*/sessions/`. Each session is a JSONL file with the full conversation history.
        """
    )
    return


if __name__ == "__main__":
    app.run()
