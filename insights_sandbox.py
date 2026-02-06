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
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Claude Code Insights Pipeline

        This notebook replicates the `/insights` command from Claude Code, teaching you about:

        - **Facet extraction** - Extracting structured data from unstructured transcripts
        - **Aggregation** - Combining individual facets into statistics
        - **Map-reduce patterns** - Running parallel LLM analysis prompts
        - **Synthesis** - Combining multiple analyses into a unified report
        """
    )
    return


@app.cell
def _(mo):
    mo.mermaid(
        """
        flowchart TD
            A[Raw Sessions] --> B[Stage 1: Facet Extraction]
            B --> C[Stage 2: Aggregate]
            C --> D[Stage 3: Map-Reduce Analysis]
            D --> E1[project_areas]
            D --> E2[interaction_style]
            D --> E3[what_works]
            D --> E4[friction_analysis]
            D --> E5[suggestions]
            D --> E6[on_the_horizon]
            D --> E7[fun_ending]
            E1 --> F[Stage 4: Synthesis]
            E2 --> F
            E3 --> F
            E4 --> F
            E5 --> F
            E6 --> F
            E7 --> F
            F --> G[HTML Report]

            style A fill:#e1f5fe
            style G fill:#c8e6c9
            style D fill:#fff3e0
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Getting Started

        1. **Get your W&B API Key**: Go to [wandb.ai/settings](https://wandb.ai/settings) and copy your API key
        2. **Configure below**: Enter your API key and choose a transcript source
        3. **Run the pipeline**: Click "Run Pipeline" to generate your report

        ### Viewing Weave Traces

        After running, visit [wandb.ai](https://wandb.ai) → your project → "Weave" tab to see all traced LLM calls.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("---\n## Configuration")
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
    mo.md(
        """
        ---
        ## Pipeline Stages

        Before running, let's understand what each stage does:
        """
    )
    return


@app.cell
def _(mo):
    mo.accordion({
        "Stage 1: Facet Extraction": mo.md("""
**What it does:** Extracts structured "facets" from each session transcript.

**What's a Facet?** A facet captures key aspects of a session:
- `underlying_goal`: What the user fundamentally wanted
- `goal_categories`: Categorized counts of user requests
- `outcome`: Whether goals were achieved (fully/mostly/partially/not)
- `satisfaction`: User satisfaction signals (happy/satisfied/frustrated)
- `friction`: What went wrong and why
- `session_type`: Nature of the interaction

**Processing notes:**
- Sessions >30KB are chunked and summarized first
- Warmup/minimal sessions are filtered out
- Requires minimum 2 user messages and 1+ minute duration
        """),

        "Stage 2: Aggregation (Pure Python)": mo.md("""
**What it does:** Combines individual facets into aggregate statistics. **No LLM needed!**

```python
# For count dictionaries:
for key, count in source.items():
    target[key] = target.get(key, 0) + count

# Top-N ranking:
top_goals = sorted(goal_categories.items(), key=lambda x: x[1], reverse=True)[:8]
```

**What gets aggregated:**
- Goal categories: Merged and ranked
- Friction types: Merged and ranked
- Outcomes: Distribution counts
- Satisfaction signals: Merged counts
- Session summaries: Collected for analysis
        """),

        "Stage 3: Map-Reduce Analysis": mo.md("""
**What it does:** Runs **7 parallel LLM prompts** that each analyze the aggregated data from a different angle.

```python
with ThreadPoolExecutor(max_workers=7) as executor:
    futures = {
        executor.submit(run_analysis_prompt, name, data): name
        for name in ANALYSIS_PROMPTS
    }
```

| Prompt | Purpose |
|--------|---------|
| `project_areas` | Identify 4-5 areas user works on |
| `interaction_style` | Describe how user interacts with Claude |
| `what_works` | Find 3 impressive workflows |
| `friction_analysis` | Categorize friction with examples |
| `suggestions` | CLAUDE.md additions, features to try |
| `on_the_horizon` | Future opportunities with copyable prompts |
| `fun_ending` | Find a memorable moment |
        """),

        "Stage 4: At-a-Glance Synthesis": mo.md("""
**What it does:** Combines all 7 analysis outputs into a unified 4-part narrative.

1. **What's working** - User's unique style and impactful accomplishments
2. **What's hindering** - Claude's fault vs user-side friction
3. **Quick wins** - Specific features to try
4. **Ambitious workflows** - What becomes possible with better models

This runs **sequentially** after all parallel prompts complete.
        """),
    })
    return


@app.cell
def _(mo):
    mo.md("---\n## Run Pipeline")
    return


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

    mo.md("---\n## Results")
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

        ## Educational Notes

        ### Why This Architecture?

        1. **Facet extraction** is the slowest step - it processes each transcript individually. Caching is important here.

        2. **Aggregation** is pure Python - fast and deterministic. No LLM needed because we're just counting and merging.

        3. **Map-reduce analysis** parallelizes well because each prompt is independent. Running 7 prompts in parallel is ~7x faster than sequential.

        4. **Synthesis** must run last because it needs all analysis outputs as input.

        ### Key Patterns

        - **Structured output**: All LLM calls use `response_format={"type": "json_object"}` for reliable parsing
        - **Prompt templates**: Prompts are inlined in the package for portability
        - **Progressive refinement**: Raw data → facets → aggregates → analysis → synthesis
        - **Weave tracing**: All LLM calls are traced for observability

        ### Finding Your Claude Code Sessions

        Sessions are stored in `~/.claude/projects/*/`. Each session is a JSONL file with the full conversation history including:
        - User messages
        - Assistant responses
        - Tool calls and results
        - Timestamps and metadata
        """
    )
    return


if __name__ == "__main__":
    app.run()
