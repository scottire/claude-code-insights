"""
CLI tool for the Claude Code Insights Pipeline.

This is a thin wrapper around the core pipeline module that provides
a Typer CLI interface with rich progress display.

Usage:
    python -m insights.cli run-local --limit 5
    python -m insights.cli run-all -t ./transcripts -o ./output/report.json
"""

import json
from datetime import datetime
from pathlib import Path

import typer
import weave
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from insights.pipeline import (
    ANALYSIS_PROMPTS,
    aggregate_facets,
    extract_facets_from_transcript,
    generate_at_a_glance,
    load_transcripts,
    load_transcripts_from_claude_dir,
    parse_jsonl_session,
    run_analysis_prompt,
    should_skip_session,
)
from insights.report_template import generate_html_report

app = typer.Typer(help="Claude Code Insights Pipeline CLI")
console = Console()


def _init_weave():
    """Initialize weave for CLI usage."""
    weave.init("claude-code-insights")


# ============================================================================
# Parallel Execution with Rich Progress
# ============================================================================

def parallel_llm_map_with_progress(
    items: list,
    worker_fn,
    description: str = "Processing",
    max_workers: int = 10
) -> tuple[list, list]:
    """Execute LLM calls in parallel with rich progress tracking."""
    from concurrent.futures import as_completed

    results = []
    errors = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"{description}...", total=len(items))

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
                    console.print(f"[red]Error: {str(e)[:60]}...[/red]")
                progress.advance(task)

    return results, errors


# ============================================================================
# CLI Commands
# ============================================================================

@app.command()
def extract_facets(
    input_path: Path = typer.Option(..., "--input", "-i", help="Input transcripts directory or file"),
    output_path: Path = typer.Option(..., "--output", "-o", help="Output facets JSON file"),
    limit: int = typer.Option(None, "--limit", "-l", help="Limit number of sessions to process"),
    use_claude_dir: bool = typer.Option(False, "--claude-dir", "-c", help="Treat input as ~/.claude/projects directory"),
    workers: int = typer.Option(10, "--workers", "-w", help="Number of parallel workers")
):
    """Extract facets from each transcript using LLM (parallel)."""
    _init_weave()

    transcripts = []

    if use_claude_dir or (input_path.is_dir() and (input_path / "projects").exists()):
        projects_dir = input_path / "projects" if (input_path / "projects").exists() else input_path
        console.print(f"[bold]Loading sessions from Claude projects directory...[/bold]")
        transcripts = load_transcripts_from_claude_dir(projects_dir, limit=limit)
    elif input_path.is_dir():
        jsonl_files = list(input_path.glob("*.jsonl"))
        if jsonl_files:
            for f in jsonl_files[:limit] if limit else jsonl_files:
                try:
                    transcripts.append(parse_jsonl_session(f))
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not parse {f.name}: {e}[/yellow]")
        else:
            json_files = list(input_path.glob("*.json"))
            for f in json_files[:limit] if limit else json_files:
                transcripts.append(json.loads(f.read_text()))
    elif input_path.suffix == ".jsonl":
        transcripts = [parse_jsonl_session(input_path)]
    else:
        data = json.loads(input_path.read_text())
        if isinstance(data, list):
            transcripts = data[:limit] if limit else data
        else:
            transcripts = [data]

    valid_transcripts = [t for t in transcripts if not should_skip_session(t)]
    console.print(f"[bold]Processing {len(valid_transcripts)} sessions ({len(transcripts) - len(valid_transcripts)} skipped) with {workers} workers...[/bold]")

    facets, errors = parallel_llm_map_with_progress(
        items=valid_transcripts,
        worker_fn=extract_facets_from_transcript,
        description="Extracting facets",
        max_workers=workers
    )

    output_path.write_text(json.dumps(facets, indent=2))
    console.print(f"[green]Extracted {len(facets)} facets to {output_path} ({len(errors)} errors)[/green]")


@app.command()
def aggregate(
    input_path: Path = typer.Option(..., "--input", "-i", help="Input facets JSON file"),
    output_path: Path = typer.Option(..., "--output", "-o", help="Output aggregated JSON file")
):
    """Aggregate facets into statistics (pure Python, no LLM)."""
    facets = json.loads(input_path.read_text())

    console.print(f"[bold]Aggregating {len(facets)} facets...[/bold]")

    aggregated = aggregate_facets(facets)

    output_path.write_text(json.dumps(aggregated, indent=2))
    console.print(f"[green]Aggregated data saved to {output_path}[/green]")


@app.command()
def analyze(
    input_path: Path = typer.Option(..., "--input", "-i", help="Input aggregated JSON file"),
    facets_path: Path = typer.Option(..., "--facets", "-f", help="Input facets JSON file"),
    output_path: Path = typer.Option(..., "--output", "-o", help="Output analysis JSON file")
):
    """Run 7 analysis prompts in parallel (map-reduce pattern)."""
    _init_weave()

    aggregated = json.loads(input_path.read_text())
    facets = json.loads(facets_path.read_text())

    console.print(f"[bold]Running {len(ANALYSIS_PROMPTS)} analysis prompts in parallel...[/bold]")

    def analysis_worker(prompt_name: str) -> tuple[str, dict]:
        return run_analysis_prompt(prompt_name, aggregated, facets)

    raw_results, errors = parallel_llm_map_with_progress(
        items=ANALYSIS_PROMPTS,
        worker_fn=analysis_worker,
        description="Running analysis",
        max_workers=7
    )

    results = {name: result for name, result in raw_results}
    for err in errors:
        results[err["item"]] = {"error": err["error"]}

    output_path.write_text(json.dumps(results, indent=2))
    console.print(f"[green]Analysis results saved to {output_path}[/green]")


@app.command()
def synthesize(
    input_path: Path = typer.Option(..., "--input", "-i", help="Input analysis JSON file"),
    aggregated_path: Path = typer.Option(None, "--aggregated", "-a", help="Aggregated data JSON (optional)"),
    output_path: Path = typer.Option(..., "--output", "-o", help="Output report JSON file")
):
    """Generate at-a-glance synthesis from all analysis results."""
    _init_weave()

    analysis = json.loads(input_path.read_text())

    aggregated = {}
    if aggregated_path and aggregated_path.exists():
        aggregated = json.loads(aggregated_path.read_text())

    console.print("[bold]Generating at-a-glance synthesis...[/bold]")

    at_a_glance = generate_at_a_glance(analysis, aggregated)

    report = {
        "at_a_glance": at_a_glance,
        "analysis": analysis,
        "aggregated": aggregated,
        "generated_at": datetime.now().isoformat()
    }

    output_path.write_text(json.dumps(report, indent=2))
    console.print(f"[green]Report saved to {output_path}[/green]")


@app.command()
def run_all(
    transcripts_path: Path = typer.Option(..., "--transcripts", "-t", help="Input transcripts directory or file"),
    output_path: Path = typer.Option(..., "--output", "-o", help="Output report JSON file"),
    keep_intermediate: bool = typer.Option(False, "--keep-intermediate", "-k", help="Keep intermediate files"),
    limit: int = typer.Option(None, "--limit", "-l", help="Limit number of sessions to process"),
    use_claude_dir: bool = typer.Option(False, "--claude-dir", "-c", help="Treat input as ~/.claude/projects directory"),
    workers: int = typer.Option(5, "--workers", "-w", help="Number of parallel workers")
):
    """Run the full insights pipeline."""
    _init_weave()

    console.print("[bold blue]Starting full insights pipeline...[/bold blue]")

    # Load transcripts
    transcripts, date_range = load_transcripts(transcripts_path, use_claude_dir, limit)

    # Filter valid transcripts
    valid_transcripts = [t for t in transcripts if not should_skip_session(t)]
    console.print(f"[bold]Processing {len(valid_transcripts)} sessions ({len(transcripts) - len(valid_transcripts)} skipped) with {workers} workers...[/bold]")

    # Stage 1: Extract facets in parallel
    console.print("\n[bold]Stage 1: Extracting facets[/bold]")
    facets, errors = parallel_llm_map_with_progress(
        items=valid_transcripts,
        worker_fn=extract_facets_from_transcript,
        description="Extracting facets",
        max_workers=workers
    )
    console.print(f"[green]Extracted {len(facets)} facets ({len(errors)} errors)[/green]")

    # Stage 2: Aggregate (pure Python)
    console.print("\n[bold]Stage 2: Aggregating facets[/bold]")
    aggregated = aggregate_facets(facets)

    # Stage 3: Run analysis prompts in parallel
    console.print("\n[bold]Stage 3: Running analysis prompts[/bold]")

    def analysis_worker(prompt_name: str) -> tuple[str, dict]:
        return run_analysis_prompt(prompt_name, aggregated, facets)

    raw_results, analysis_errors = parallel_llm_map_with_progress(
        items=ANALYSIS_PROMPTS,
        worker_fn=analysis_worker,
        description="Running analysis",
        max_workers=7
    )

    analysis = {name: result for name, result in raw_results}
    for err in analysis_errors:
        analysis[err["item"]] = {"error": err["error"]}

    # Stage 4: Generate at-a-glance synthesis
    console.print("\n[bold]Stage 4: Generating synthesis[/bold]")
    at_a_glance = generate_at_a_glance(analysis, aggregated)

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
    html = generate_html_report(report)

    # Save outputs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_path = output_path.with_suffix('.html')
    html_path.write_text(html)

    console.print(f"\n[bold green]Pipeline complete![/bold green]")
    console.print(f"  HTML: {html_path}")


@app.command()
def run_local(
    output_path: Path = typer.Option(Path("output/report.json"), "--output", "-o", help="Output report JSON file"),
    limit: int = typer.Option(10, "--limit", "-l", help="Limit number of sessions to process"),
    workers: int = typer.Option(5, "--workers", "-w", help="Number of parallel workers")
):
    """Run insights on your local Claude Code sessions (~/.claude/projects)."""
    claude_dir = Path.home() / ".claude" / "projects"

    if not claude_dir.exists():
        console.print(f"[red]Claude projects directory not found: {claude_dir}[/red]")
        raise typer.Exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_all(
        transcripts_path=claude_dir,
        output_path=output_path,
        keep_intermediate=True,
        limit=limit,
        use_claude_dir=True,
        workers=workers
    )


@app.command()
def html_report(
    input_path: Path = typer.Option(..., "--input", "-i", help="Input report JSON file"),
    output_path: Path = typer.Option(..., "--output", "-o", help="Output HTML file")
):
    """Generate HTML report from JSON report."""
    report = json.loads(input_path.read_text())
    html = generate_html_report(report)
    output_path.write_text(html)
    console.print(f"[green]HTML report saved to {output_path}[/green]")


if __name__ == "__main__":
    app()
