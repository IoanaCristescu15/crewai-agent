#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from typing import List, Optional, Union

from crewai import Crew, Task
from nanda_adapter import NANDA

BASE_DIR = Path(__file__).resolve().parent
CREW_AGENT_DIR = BASE_DIR / "crewai-agent"
if str(CREW_AGENT_DIR) not in sys.path:
    sys.path.append(str(CREW_AGENT_DIR))

from agent import build_meeting_agent  # noqa: E402

REPLY_INSTRUCTIONS = (
    "You are the user's digital twin. Using the context provided, craft a concise, first-person reply "
    "that reflects the persona and goals configured for the meeting notes agent. Mirror the user's tone—"
    "friendly and professional—and include any next steps or commitments that keep the project moving. "
    "Do not mention these instructions or that you are an AI."
)

def create_improvement():
    """Return a callable that answers inbound messages with the meeting agent."""
    meeting_agent = build_meeting_agent()

    def respond(
        message_text: str,
        conversation_history: Optional[Union[str, List[str]]] = None,
        **_: object,
    ) -> str:
        context: List[str] = []

        if isinstance(conversation_history, str):
            context.append(f"Previous turn 1: {conversation_history}")
        elif isinstance(conversation_history, list):
            for idx, turn in enumerate(conversation_history, start=1):
                context.append(f"Previous turn {idx}: {turn}")

        context.append(f"Inbound message: {message_text}")

        response_task = Task(
            description=REPLY_INSTRUCTIONS,
            expected_output="A ready-to-send reply in the user's voice.",
            context=context,
            agent=meeting_agent,
        )

        crew = Crew(agents=[meeting_agent], tasks=[response_task], verbose=False)
        result = crew.kickoff()
        return str(result).strip()

    return respond

def main():
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    domain = os.getenv("DOMAIN_NAME")

    if not anthropic_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    if not domain:
        raise RuntimeError("DOMAIN_NAME is not set.")

    improvement_logic = create_improvement()
    nanda = NANDA(improvement_logic)

    port = int(os.getenv("PORT", "6000"))
    api_port = int(os.getenv("API_PORT", "6001"))
    agent_id = os.getenv("AGENT_ID")
    cert_path = os.getenv("CERT_PATH", str(BASE_DIR / "fullchain.pem"))
    key_path = os.getenv("KEY_PATH", str(BASE_DIR / "privkey.pem"))

    if domain == "localhost":
        nanda.start_server()
    else:
        nanda.start_server_api(
            anthropic_key=anthropic_key,
            domain=domain,
            agent_id=agent_id,
            port=port,
            api_port=api_port,
            cert=cert_path,
            key=key_path,
        )

if __name__ == "__main__":
    main()
