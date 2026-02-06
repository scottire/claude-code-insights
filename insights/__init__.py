"""
Claude Code Insights Pipeline.

A tool for analyzing Claude Code session transcripts and generating
insight reports about usage patterns, friction points, and suggestions.
"""

from insights.pipeline import (
    init_weave,
    get_weave_url,
    get_client,
    run_insights_pipeline,
    load_transcripts,
    parse_jsonl_session,
)
from insights.report_template import generate_html_report

__all__ = [
    "init_weave",
    "get_weave_url",
    "get_client",
    "run_insights_pipeline",
    "load_transcripts",
    "parse_jsonl_session",
    "generate_html_report",
]
