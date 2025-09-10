from __future__ import annotations

import argparse
import os
import sys
from typing import List

from crewai import Crew, Task

from agent import (
    build_meeting_agent, build_coding_agent,
    create_meeting_tasks, create_coding_tasks
)
from tools import UrlReaderTool, PdfReaderTool, PasteTool


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Digital Twin Assistant - comprehensive AI agent for various tasks."
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["meeting", "coding"],
        default="meeting",
        help="Select agent mode: meeting (notes) or coding"
    )
    
    # Meeting mode arguments
    parser.add_argument("--url", type=str, help="Exact URL to fetch", default=None)
    parser.add_argument("--pdf", type=str, help="Local PDF path", default=None)
    parser.add_argument("--text", type=str, help="Raw pasted text", default=None)
    parser.add_argument(
        "--weekly",
        action="store_true",
        help="Weekly mode: accept multiple sources and produce one combined digest",
    )
    
    # Coding mode arguments
    parser.add_argument("--code", type=str, help="Code to analyze for coding mode", default=None)
    parser.add_argument("--code-file", type=str, help="File containing code to analyze", default=None)
    
    return parser.parse_args(argv)


def ensure_api_key():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)


def run_introduction(agent):
    task = Task(
        description=(
            "Introduce yourself to the class in 3 sentences as my digital twin. "
            "Output exactly 3 sentences, first-person."
        ),
        expected_output="Exactly 3 sentences, first-person.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()
    print(result)


def run_coding_mode(code: str):
    """Run coding mode with specified code."""
    agent = build_coding_agent()
    analyze_task, explain_task = create_coding_tasks(code, agent)
    
    crew = Crew(agents=[agent], tasks=[analyze_task, explain_task])
    result = crew.kickoff()
    print(result)




def print_template():
    template = (
        "1) TL;DR:\n"
        "- [example bullet]\n"
        "- [example bullet]\n"
        "- [example bullet]\n\n"
        "2) Decisions:\n"
        "- [not found]\n\n"
        "3) Risks/Blockers:\n"
        "- Reliability: [not found]\n"
        "- Latency: [not found]\n"
        "- Cost: [not found]\n"
        "- Rollout safety: [not found]\n\n"
        "4) Next Steps:\n"
        "- Task: [not found] | Owner: [not found] | Due: [not found]"
    )
    print(template)


def gather_sources(args: argparse.Namespace) -> List[str]:
    url_tool = UrlReaderTool()
    pdf_tool = PdfReaderTool()
    paste_tool = PasteTool()

    sources: List[str] = []

    if args.url:
        sources.append(url_tool.run(args.url) or "")
    if args.pdf:
        sources.append(pdf_tool.run(args.pdf) or "")
    if args.text:
        sources.append(paste_tool.run(args.text) or "")

    return [s for s in sources if s]


def summarize_single(agent, source_text: str):
    # Include the source text in the description instead of using context
    task = Task(
        description=(
            f"Given the following source text, output the four sections in order with "
            "strict formatting and [not found] placeholders when needed.\n\n"
            f"SOURCE TEXT:\n{source_text}\n\n"
            "1) TL;DR: exactly 3 bullets\n"
            "2) Decisions: bullets\n"
            "3) Risks/Blockers: bullets (consider reliability, latency, cost, rollout safety)\n"
            "4) Next Steps: bullets with Owner: and Due: when present, else [not found]"
        ),
        expected_output="Four sections in order with strict formatting.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()
    print(result)


def weekly_rollup(agent, sources: List[str]):
    if len(sources) < 2:
        print("Hint: --weekly needs at least two sources (mix of --url/--pdf/--text).")
        return
    
    # Combine all sources into the description
    sources_text = "\n\n".join([f"SOURCE {i+1}:\n{source}" for i, source in enumerate(sources)])
    
    task = Task(
        description=(
            f"Given the following {len(sources)} sources, merge them into a single digest with the four sections:\n\n"
            f"{sources_text}\n\n"
            "1) TL;DR: exactly 3 bullets\n"
            "2) Decisions: bullets\n"
            "3) Risks/Blockers: bullets (consider reliability, latency, cost, rollout safety)\n"
            "4) Next Steps: bullets with Owner: and Due: when present, else [not found]"
        ),
        expected_output="Merged four-section digest.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()
    print(result)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    ensure_api_key()

    # Handle different modes
    if args.mode == "coding":
        code = None
        if args.code:
            code = args.code
        elif args.code_file:
            try:
                with open(args.code_file, 'r') as f:
                    code = f.read()
            except FileNotFoundError:
                print(f"Error: File {args.code_file} not found.", file=sys.stderr)
                return 1
        else:
            print("Error: --code or --code-file is required for coding mode.", file=sys.stderr)
            return 1
        
        run_coding_mode(code)
        return 0


    elif args.mode == "meeting":
        # Original meeting notes functionality
        agent = build_meeting_agent()

        if not args.url and not args.pdf and not args.text:
            # No source: run intro and print template
            run_introduction(agent)
            print()
            print_template()
            return 0

        if args.weekly:
            sources = gather_sources(args)
            weekly_rollup(agent, sources)
            return 0

        # Single-source mode: enforce exactly one
        provided = [x for x in [args.url, args.pdf, args.text] if x]
        if len(provided) != 1:
            print("Error: Provide exactly one of --url, --pdf, or --text (or use --weekly).", file=sys.stderr)
            return 2

        sources = gather_sources(args)
        if not sources:
            print("Error: Source could not be read.", file=sys.stderr)
            return 3

        summarize_single(agent, sources[0])
        return 0

    else:
        print(f"Error: Unknown mode '{args.mode}'.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


