from __future__ import annotations

import os
from typing import List

from crewai import Agent, Task, Crew, LLM

from tools import (
    UrlReaderTool, PdfReaderTool, PasteTool, WebSearchTool, 
    CodeAnalysisTool
)


PERSONA_BACKSTORY = (
    "You are a second-year Master's in Data Science student at Harvard University, "
    "originally from Romania, with a B.S. in Computer Science & Mathematics from the "
    "University of Richmond. You enjoy travel and extreme activities like scuba diving, "
    "skiing, skydiving, and swimming with sharks. Your professional focus is ML infrastructure."
)

GOAL_TEMPLATE = (
    "1) TL;DR: exactly 3 bullets\n"
    "2) Decisions: bullets\n"
    "3) Risks/Blockers: bullets (consider reliability, latency, cost, rollout safety)\n"
    "4) Next Steps: bullets with Owner: and Due: when present, else [not found]"
)

SYSTEM_GUARDRAILS = (
    "Tone: diplomatic, happy, respectful. Avoid repetition and unusual words. "
    "Never use em dashes. Do not invent facts; if data is missing, write [not found]. "
    "Refuse or escalate legal/finance, HR-sensitive, and security-sensitive topics."
)


def build_meeting_agent() -> Agent:
    """Agent specialized in meeting notes and documentation."""
    llm = LLM(model="anthropic/claude-3-5-sonnet-20240620")
    tools = [UrlReaderTool(), PdfReaderTool(), PasteTool()]

    return Agent(
        role="Meeting Notes Scribe",
        goal=(
            "Turn meeting inputs into concise, actionable notes for an engineer "
            "who is a Harvard DS master's student from Romania with a CS/Math background.\n"
            + GOAL_TEMPLATE
        ),
        backstory=(
            "You compress technical context into clear decisions and next steps suitable for "
            "class projects, research meetings, and infra reviews. You're practical and concise. "
            + PERSONA_BACKSTORY
        ),
        allow_delegation=False,
        verbose=False,
        llm=llm,
        tools=tools,
    )




def build_coding_agent() -> Agent:
    """Agent specialized in code analysis and development assistance."""
    llm = LLM(model="anthropic/claude-3-5-sonnet-20240620")
    tools = [CodeAnalysisTool(), WebSearchTool(), PasteTool()]

    return Agent(
        role="Code Review Assistant",
        goal=(
            "Analyze code for bugs, performance issues, and suggest improvements. "
            "Provide clear explanations of code functionality and best practices."
        ),
        backstory=(
            "You're a skilled software engineer with expertise in Python, data science libraries, "
            "and ML infrastructure. You have a keen eye for code quality, security issues, "
            "and performance optimization. You communicate technical concepts clearly. "
            + PERSONA_BACKSTORY
        ),
        allow_delegation=False,
        verbose=False,
        llm=llm,
        tools=tools,
    )






def create_meeting_tasks(source_texts: List[str], agent: Agent):
    """Create tasks for meeting notes functionality."""
    introduce_task = Task(
        description=(
            "Introduce yourself to the class in 3 sentences as my digital twin. "
            "Output exactly 3 sentences, first-person."
        ),
        expected_output="Exactly 3 sentences, first-person.",
        agent=agent,
    )

    summarize_task = Task(
        description=(
            "Given one source (URL, PDF, or pasted text), output the four sections in order with "
            "strict formatting and [not found] placeholders when needed.\n" + GOAL_TEMPLATE
        ),
        expected_output="Four sections in order with strict formatting.",
        context=source_texts[:1] if source_texts else None,
        agent=agent,
    )

    weekly_task = Task(
        description=(
            "Given two or more sources, merge them into a single digest with the four sections. "
            "If fewer than two sources, return a helpful usage hint.\n" + GOAL_TEMPLATE
        ),
        expected_output="Merged four-section digest or usage hint if <2 sources.",
        context=source_texts if source_texts else None,
        agent=agent,
    )

    return introduce_task, summarize_task, weekly_task




def create_coding_tasks(code: str, agent: Agent):
    """Create tasks for code analysis functionality."""
    analyze_task = Task(
        description=(
            f"Analyze the following code for bugs, performance issues, and improvements:\n\n{code}\n\n"
            "Provide a detailed analysis including language detection, potential issues, "
            "and specific suggestions for improvement."
        ),
        expected_output="Comprehensive code analysis with issues and suggestions.",
        agent=agent,
    )

    explain_task = Task(
        description=(
            f"Explain what the following code does in simple terms:\n\n{code}\n\n"
            "Break down the functionality, data flow, and main components. "
            "Make it accessible to someone learning programming."
        ),
        expected_output="Clear explanation of code functionality and components.",
        agent=agent,
    )

    return analyze_task, explain_task






def create_tasks(source_texts: List[str], agent: Agent):
    """Legacy function for backward compatibility."""
    return create_meeting_tasks(source_texts, agent)


def build_crew() -> Crew:
    agent = build_meeting_agent()
    introduce_task, summarize_task, weekly_task = create_tasks([], agent)
    return Crew(agents=[agent], tasks=[introduce_task, summarize_task, weekly_task])


