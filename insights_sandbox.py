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

An educational notebook that teaches the /insights pipeline architecture
while also letting you run it on your own Claude Code sessions.

Run with:
    uvx marimo edit --sandbox insights_sandbox.py

Or from GitHub:
    uvx marimo edit --sandbox https://raw.githubusercontent.com/scottire/claude-code-insights/main/insights_sandbox.py
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    return mo, Path


@app.cell
def _(mo):
    # Configuration inputs
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
    local_limit = mo.ui.slider(
        label="Sessions to analyze",
        start=5,
        stop=50,
        value=10,
        step=5,
    )
    run_button = mo.ui.run_button(label="Run Pipeline", full_width=True)

    return api_key_input, local_limit, weave_project_input, workers_input, run_button


@app.cell
def _(Path, local_limit):
    # Load transcripts from local Claude directory
    claude_dir = Path.home() / ".claude" / "projects"
    transcripts = []
    load_error = None

    if not claude_dir.exists():
        load_error = f"Claude projects directory not found: {claude_dir}"
    else:
        from insights.pipeline import load_transcripts_from_claude_dir
        transcripts = load_transcripts_from_claude_dir(claude_dir, limit=local_limit.value)

    return claude_dir, transcripts, load_error, load_transcripts_from_claude_dir


@app.cell
def _(
    api_key_input,
    load_error,
    local_limit,
    mo,
    run_button,
    transcripts,
    weave_project_input,
    workers_input,
):
    # Sidebar with configuration
    sidebar_content = mo.vstack([
        mo.md("## Configuration"),
        api_key_input,
        weave_project_input,
        mo.md("---"),
        mo.md("## Sessions"),
        local_limit,
        workers_input,
        mo.md(f"**Found {len(transcripts)} sessions**") if not load_error else mo.callout(load_error, kind="danger"),
        mo.md("---"),
        run_button,
        mo.md("---"),
        mo.md("""
### Pipeline Stages
1. **Facet Extraction** - Extract structured data from each session
2. **Aggregation** - Combine into statistics (no LLM)
3. **Map-Reduce** - 7 parallel analysis prompts
4. **Synthesis** - Generate at-a-glance summary
        """),
    ])

    return (sidebar_content,)


@app.cell
def _(mo):
    # Main content header
    mo.md(
        r"""
        # Claude Code Insights Pipeline

        This notebook replicates the `/insights` command from Claude Code. Configure your settings in the sidebar and click **Run Pipeline** to analyze your sessions.
        """
    )
    return


@app.cell
def _(mo):
    mo.mermaid(
        """
        flowchart LR
            A[Sessions] --> B[Facets]
            B --> C[Aggregate]
            C --> D[7 Parallel Prompts]
            D --> E[Synthesis]
            E --> F[HTML Report]

            style A fill:#e1f5fe
            style F fill:#c8e6c9
            style D fill:#fff3e0
        """
    )
    return


@app.cell
def _(
    api_key_input,
    mo,
    run_button,
    transcripts,
    weave_project_input,
    workers_input,
):
    # Pipeline execution
    mo.stop(not run_button.value, mo.md("*Click 'Run Pipeline' in the sidebar to start*"))

    if not api_key_input.value:
        mo.stop(True, mo.callout("Please enter your W&B API Key in the sidebar.", kind="warn"))

    if not transcripts:
        mo.stop(True, mo.callout("No transcripts found to analyze.", kind="warn"))

    from insights.pipeline import init_weave, run_insights_pipeline
    import weave

    # Initialize Weave
    init_weave(weave_project_input.value)

    # Get the weave trace URL base
    weave_url = f"https://wandb.ai/{weave_project_input.value}/weave"

    mo.md(f"**Starting pipeline...** Traces will appear at: [{weave_url}]({weave_url})")

    return init_weave, run_insights_pipeline, weave, weave_url


@app.cell
def _(
    api_key_input,
    mo,
    run_button,
    run_insights_pipeline,
    transcripts,
    weave_url,
    workers_input,
):
    mo.stop(not run_button.value)

    # Run the pipeline
    with mo.status.spinner("Running pipeline...") as _spinner:
        html_bytes = run_insights_pipeline(
            transcripts=transcripts,
            api_key=api_key_input.value,
            workers=workers_input.value,
        )

    html_report = html_bytes.decode("utf-8")

    mo.vstack([
        mo.callout(
            mo.md(f"**Pipeline complete!** View traces at: [{weave_url}]({weave_url})"),
            kind="success"
        ),
    ])

    return html_bytes, html_report


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
    ], justify="start", gap=2)

    return (download,)


@app.cell
def _(html_report, mo, run_button):
    mo.stop(not run_button.value)
    mo.stop(not html_report)

    # Show the HTML report inline
    mo.Html(html_report)
    return


@app.cell
def _(mo, sidebar_content):
    # Layout with sidebar
    mo.sidebar(sidebar_content)
    return


if __name__ == "__main__":
    app.run()
