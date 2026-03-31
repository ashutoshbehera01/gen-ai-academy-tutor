# gen-ai-academy-tutor
A multi-agent AI Tutor built with Gemini 2.5 Flash and Google ADK for Hack2Skill Track 1.

## Architecture
User → Greeter Agent → SequentialAgent Workflow
                              ↓
                  Researcher Agent (Wikipedia Tool)
                              ↓
                     Response Formatter Agent

## Tech Stack
- Google Agent Development Kit (ADK)
- Gemini 2.5 Flash via Vertex AI
- LangChain + Wikipedia API
- Python 3.12

## Features
- Multi-agent SequentialAgent architecture
- Wikipedia Tool Calling via LangChain
- State management with add_prompt_to_state
- Cloud Logging via Google Cloud
