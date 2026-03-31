import os
import logging
import google.cloud.logging
from dotenv import load_dotenv

from google.adk import Agent
from google.adk.agents import SequentialAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.langchain_tool import LangchainTool

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

import google.auth
import google.auth.transport.requests
import google.oauth2.id_token

# --- Setup Logging and Environment ---
cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

load_dotenv()

model_name = os.getenv("MODEL")

# --- Greet user and save their prompt ---
def add_prompt_to_state(
    tool_context: ToolContext, prompt: str
) -> dict[str, str]:
    """Saves the user's initial prompt to the state."""
    tool_context.state["PROMPT"] = prompt
    logging.info(f"[State updated] Added to PROMPT: {prompt}")
    return {"status": "success"}

# --- Configuring the Wikipedia tool ---
wikipedia_tool = LangchainTool(
    tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
)

# 1. Researcher Agent
comprehensive_researcher = Agent(
    name="comprehensive_researcher",
    model=model_name,
    description="The primary researcher that can access Wikipedia to find accurate information.",
    instruction="""
    You are a helpful research assistant. Your goal is to fully answer the user's PROMPT.

    You have access to a Wikipedia tool to search for accurate definitions and facts about AI, programming, and tech topics.

    First, analyze the user's PROMPT.
    - If the prompt can be answered by only one tool, use just that tool.
    - If the prompt is complex and requires information from Wikipedia, you MUST use the tool.
    - Synthesize the results from the tool(s) you use into preliminary data outputs.

    PROMPT:
    { PROMPT }
    """,
    tools=[
        wikipedia_tool
    ],
    output_key="research_data"
)

# 2. Response Formatter Agent
response_formatter = Agent(
    name="response_formatter",
    model=model_name,
    description="Synthesizes all information in a friendly, readable response.",
    instruction="""
    You are the friendly voice of the AI Tutor. Your task is to take the
    RESEARCH_DATA and present it to the user in a complete and helpful answer.

    - First, present the key information (topic name, definition, and important facts).
    - Then, add interesting context or background from the research.
    - If some information is missing, just present what you have.
    - Be conversational, clear, and encouraging for students.

    RESEARCH_DATA:
    { research_data }
    """
)

tutor_workflow = SequentialAgent(
    name="tutor_workflow",
    description="The sequential workflow for answering a user's question about AI or tech.",
    sub_agents=[
        comprehensive_researcher,   # Step 1: Gather all data
        response_formatter,          # Step 2: Format the final response
    ]
)

root_agent = Agent(
    name="greeter",
    model=model_name,
    description="The main entry point for the AI Tutor.",
    instruction="""
    You are the 'Gen AI Academy Tutor.'
    - Let the user know you will help them learn about AI and technology.
    - When the user responds, use the `add_prompt_to_state` tool to save their response.
    After using the tool, transfer control to the `tutor_workflow` agent.
    """,
    tools=[add_prompt_to_state],
    sub_agents=[tutor_workflow]
)