# Digital Twin Assistant (CrewAI)

**Author:** Ioana Cristescu <br>
**Track:** Tech Track

## Project Report

### Overview
Digital Twin Assistant is a CrewAI project that represents a second-year Master's in Data Science student at Harvard University from Romania, with a B.S. in Computer Science & Mathematics from the University of Richmond. The agent system provides specialized capabilities for meeting notes and code analysis.

### Test Results

#### Test Prompts Executed
**Prompt**: "Introduce yourself to the class" <br>
**Command**: `python -m cli --mode meeting` <br>
**Result**: **SUCCESS** <br>

```
I'm a second-year Master's in Data Science student at Harvard University, originally from Romania, with a strong background in Computer Science and Mathematics from the University of Richmond. My professional focus is on machine learning infrastructure, where I'm passionate about building robust and efficient systems. In my free time, I'm an adventure enthusiast, enjoying extreme activities like scuba diving, skiing, skydiving, and even swimming with sharks.
```

**Prompt**: "Explain my background in 3 sentences" <br>
**Command**: `python -m cli --mode meeting --text "Please explain my background in exactly 3 sentences"` <br>
**Result**: **SUCCESS** <br>
- Properly structured in TL;DR format
- Captured key academic and personal details
- Maintained consistent personality

### Implementation Analysis

#### What Works Well
- **Focused Agent Design**: Single-purpose agents (meeting, coding) are highly reliable and consistent
- **Tool Integration**: Well-designed tools provide consistent functionality and good error handling
- **Educational Value**: Code analysis provides excellent learning-oriented explanations with security awareness
- **Persona Consistency**: Maintains unique Harvard DS student personality across all modes
- **Structured Output**: Meeting notes follow consistent 4-section format (TL;DR, Decisions, Risks/Blockers, Next Steps)
- **Input Flexibility**: Works with text, URLs, and PDFs
- **Error Handling**: Graceful degradation with `[not found]` placeholders

#### Areas for Improvement
- **Weekly Mode**: Doesn't recognize multiple `--text` arguments as separate sources
- **Error Messages**: More specific error messages for invalid inputs would improve user experience
- **Output Length**: Code analysis explanations are very verbose and could be more concise
- **Formatting Consistency**: Meeting notes sometimes have inconsistent formatting in the structured output
- **Input Validation**: Better validation for empty or invalid inputs

#### Technical Strengths
- **CrewAI Framework**: Works excellently for focused, single-purpose agents
- **Tool Design**: Tools are well-architected with proper error handling and fallbacks
- **CLI Design**: Clean argument parsing and helpful error messages for missing arguments
- **Code Quality**: Clean, maintainable code with good separation of concerns

### Key Learnings
1. **Agent Design**: Specialized agents with focused tools perform better than general-purpose agents
2. **Tool Selection**: Choose tools that complement each other and cover the agent's domain thoroughly
3. **Error Handling**: Robust error handling is crucial for external API integrations
4. **User Experience**: Clear CLI modes and helpful error messages improve usability significantly
5. **Quality over Quantity**: Better to have fewer, fully working modes than many partially working ones
6. **Testing**: Comprehensive testing revealed which modes work reliably and which don't

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- Anthropic API key

### Installation Steps

1) Create and activate a virtual environment
- macOS/Linux:
```bash
python3 -m venv .venv && source .venv/bin/activate
```
- Windows (PowerShell):
```bash
py -m venv .venv; .venv\Scripts\Activate.ps1
```

2) Install dependencies
```bash
pip install -r requirements.txt
```

3) Configure Anthropic API key
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```
Windows (PowerShell):
```powershell
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

## Usage

### Meeting Notes Mode
```bash
# Introduction and template
python -m cli --mode meeting

# Single source
python -m cli --mode meeting --url https://example.com/page
python -m cli --mode meeting --pdf /path/to/file.pdf
python -m cli --mode meeting --text "raw notes here"

# Weekly rollup (≥2 sources, mixed allowed)
python -m cli --mode meeting --weekly --url https://a --pdf /path/to/b.pdf --text "snippet"
```

### Voice Mode (Speech I/O)
```bash
# Interactive recording loop: speak meeting notes, hear the TL;DR back
python -m cli --mode meeting --voice

# Transcribe an existing audio file and summarize it
python -m cli --mode meeting --input-audio /path/to/notes.wav

# Save synthesized audio and reuse a specific TTS voice/rate
python -m cli --mode meeting --voice --response-audio reply.wav --tts-voice "com.apple.speech.synthesis.voice.samantha"
```

- Whisper STT model defaults to `base`; override with `--stt-model small`.
- `--no-playback` skips immediate audio playback (helpful on remote servers).
- Use `--keep-recordings` to retain temporary microphone captures.

> **System packages:** Speech features need PortAudio. On macOS: `brew install portaudio`. On Debian/Ubuntu: `sudo apt-get install portaudio19-dev`.

### Coding Mode
```bash
# Analyze code directly
python -m cli --mode coding --code "def hello(): print('world')"

# Analyze code from file
python -m cli --mode coding --code-file /path/to/script.py
```

## Output Format

### Meeting Notes Output
```
1) TL;DR:
- Brief point one
- Brief point two
- Brief point three

2) Decisions:
- [not found]

3) Risks/Blockers:
- Reliability: [not found]
- Latency: [not found]
- Cost: [not found]
- Rollout safety: [not found]

4) Next Steps:
- Task: [not found] | Owner: [not found] | Due: [not found]
```

## Agent Capabilities

### Specialized Agents
- **Meeting Notes Scribe**: Processes meeting inputs into structured, actionable notes
- **Code Review Assistant**: Analyzes code for issues and provides improvement suggestions

### Additional Components
- **Nanda Provider (`digital_twin_nanda.py`)**: Exposes the meeting agent through the NANDA server so the digital twin can answer inbound messages; set the Anthropic credentials plus `DOMAIN_NAME`, optional TLS paths, and run the script to launch the provider.

### Agent + Adapter Overview
The agent turns meeting inputs into a short, first-person reply that sounds like the student. It adds clear next steps and keeps the tone consistent. It can read prior conversation turns to stay aligned with context. We expose the agent through the NANDA provider by running `digital_twin_nanda.py`. The script serves an HTTP endpoint. It accepts a message and optional history and returns a reply.

### Adapter Feedback
Some README instructions are confusing. The environment variables are listed in different places, and TLS setup is not grouped. The difference between localhost and domain modes is easy to miss. I also had trouble seeing the agent on the website. Aditionally, a curl example and a basic health endpoint would make verification easier. 

### Tools Available
- **URL Reader**: Fetches and extracts text from web pages
- **PDF Reader**: Extracts text from local PDF files
- **Web Search**: Searches the web for current information (DuckDuckGo API)
- **Code Analysis**: Analyzes code for bugs, performance, and improvements
- **Paste Tool**: Handles raw text input

## Guardrails & Limitations
- Diplomatic, happy, respectful tone; avoid repetition and unusual words
- Never use em dashes
- Do not invent facts; if data is missing, write `[not found]`
- Refuse/escalate legal/finance, HR-sensitive, and security-sensitive topics
- Web search uses DuckDuckGo API (no API key required) for current information
