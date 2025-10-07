from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

from crewai import Crew, Task

from agent import (
    build_meeting_agent, build_coding_agent,
    create_meeting_tasks, create_coding_tasks
)
from tools import UrlReaderTool, PdfReaderTool, PasteTool
from audio_io import VoiceIO


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
    
    # Voice mode arguments
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Enable interactive voice loop (records speech, runs the agent, plays audio response).",
    )
    parser.add_argument(
        "--input-audio",
        type=str,
        help="Path to a pre-recorded audio file; will be transcribed and used as text input.",
        default=None,
    )
    parser.add_argument(
        "--response-audio",
        type=str,
        help="Optional path to save synthesized agent responses when using voice features.",
        default=None,
    )
    parser.add_argument(
        "--stt-model",
        type=str,
        default="base",
        help="Whisper model name/path to load for speech-to-text (default: base).",
    )
    parser.add_argument(
        "--tts-voice",
        type=str,
        help="Optional pyttsx3 voice identifier to use for text-to-speech.",
        default=None,
    )
    parser.add_argument(
        "--tts-rate",
        type=int,
        help="Optional speech rate override for text-to-speech (pyttsx3).",
        default=None,
    )
    parser.add_argument(
        "--no-playback",
        action="store_true",
        help="Skip audio playback when synthesizing responses (useful on headless systems).",
    )
    parser.add_argument(
        "--keep-recordings",
        action="store_true",
        help="Keep temporary microphone recordings instead of deleting them after processing.",
    )
    
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


def summarize_single(agent, source_text: str, *, display: bool = True) -> str:
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
    text_result = str(result)
    if display:
        print(text_result)
    return text_result


def weekly_rollup(agent, sources: List[str], *, display: bool = True) -> str | None:
    if len(sources) < 2:
        print("Hint: --weekly needs at least two sources (mix of --url/--pdf/--text).")
        return None
    
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
    text_result = str(result)
    if display:
        print(text_result)
    return text_result


def run_meeting_voice_session(agent, args, voice_io: VoiceIO) -> None:
    """Interactive loop that records speech, summarizes it, and plays audio responses."""
    print("Voice session ready.")
    print("Say your meeting notes after the prompt. Say 'exit' to leave.\n")

    while True:
        try:
            transcript, audio_path = voice_io.capture_and_transcribe()
        except ImportError as exc:
            print(f"Missing dependency: {exc}", file=sys.stderr)
            return
        except RuntimeError as exc:
            print(f"Recorder error: {exc}", file=sys.stderr)
            continue

        transcript_text = transcript.strip()
        input_audio_path = Path(audio_path)

        if not transcript_text:
            print("No speech detected. Please try again.")
            if not args.keep_recordings:
                input_audio_path.unlink(missing_ok=True)
            continue

        if transcript_text.lower() in {"exit", "quit", "stop"}:
            print("Ending voice session.")
            if not args.keep_recordings:
                input_audio_path.unlink(missing_ok=True)
            else:
                print(f"[saved] Input audio: {input_audio_path}")
            break

        print("\nTranscription:")
        print(transcript_text)

        summary = summarize_single(agent, transcript_text, display=False)
        print("\nAgent response:")
        print(summary)

        response_path: Path | None = None
        try:
            response_path = voice_io.synthesize_speech(
                summary,
                Path(args.response_audio) if args.response_audio else None,
            )
            if args.response_audio:
                response_path = Path(args.response_audio)
            if not args.no_playback:
                voice_io.play_audio(response_path)
        except ImportError as exc:
            print(f"Missing dependency for TTS: {exc}", file=sys.stderr)
        except RuntimeError as exc:
            print(f"TTS error: {exc}", file=sys.stderr)

        if args.keep_recordings:
            print(f"[saved] Input audio: {input_audio_path}")
            if response_path:
                print(f"[saved] Response audio: {response_path}")
        else:
            input_audio_path.unlink(missing_ok=True)
            if response_path and not args.response_audio:
                response_path.unlink(missing_ok=True)
            elif response_path:
                print(f"[saved] Response audio: {response_path}")

        print("\n---\n")


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    ensure_api_key()

    voice_io: VoiceIO | None = None
    if args.voice or args.input_audio:
        voice_io = VoiceIO(
            stt_model_name=args.stt_model,
            tts_voice=args.tts_voice,
            tts_rate=args.tts_rate,
        )

    # Handle different modes
    if args.mode == "coding":
        if args.voice or args.input_audio:
            print("Voice features are currently available for meeting mode only.", file=sys.stderr)
            return 1

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

        if args.voice:
            if voice_io is None:
                voice_io = VoiceIO(
                    stt_model_name=args.stt_model,
                    tts_voice=args.tts_voice,
                    tts_rate=args.tts_rate,
                )
            run_meeting_voice_session(agent, args, voice_io)
            return 0

        if args.input_audio:
            assert voice_io is not None  # For type-checkers; instantiated above.
            try:
                transcript = voice_io.transcribe_file(args.input_audio)
            except ImportError as exc:
                print(f"Missing dependency for transcription: {exc}", file=sys.stderr)
                return 4
            except RuntimeError as exc:
                print(f"Transcription error: {exc}", file=sys.stderr)
                return 4

            if not transcript.strip():
                print("Error: Transcription produced empty text.", file=sys.stderr)
                return 5

            print("Transcribed audio input:")
            print(transcript)
            args.text = transcript

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
