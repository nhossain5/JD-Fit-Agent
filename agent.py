"""
JD Fit Agent -- Milestone 1
A minimal agentic loop: an LLM that calls ONE tool to do its job.

This is deliberately small. It proves the core mechanic behind every agent:
  1. The model receives a task.
  2. It DECIDES to call a tool.
  3. We run the tool and feed the result back.
  4. The model uses that result to produce a final answer.

Runtime model: Google Gemini (free tier), via its OpenAI-compatible endpoint.
Writing against the OpenAI SDK means you can later swap to Groq, OpenAI, etc.
by changing only BASE_URL, MODEL, and the API key -- nothing else.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Model configuration -----------------------------------------------------
# Gemini's OpenAI-compatible endpoint. To switch providers later, change only
# these three values (e.g., Groq: https://api.groq.com/openai/v1 + a Llama model).
# BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "openai/gpt-oss-120b"
# API_KEY = os.environ.get("GEMINI_API_KEY")
API_KEY = os.environ.get("GROQ_API_KEY")

if not API_KEY:
    raise SystemExit("Set API_KEY in your .env file (copy .env.example).")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# --- The one tool ------------------------------------------------------------
def find_missing_keywords(resume_text: str, required_keywords: list) -> dict:
    """Deterministically check which required keywords appear in the resume.

    Plain Python -- no LLM here. The *agent* decides WHEN to call this;
    the tool itself just returns facts the model can reason over.
    """
    text = resume_text.lower()
    present, missing = [], []
    for kw in required_keywords:
        (present if kw.lower() in text else missing).append(kw)
    return {"present": present, "missing": missing}


# Tool schema the model sees (OpenAI function-calling format).
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_missing_keywords",
            "description": "Check which required skills/keywords are present or missing in a resume.",
            "parameters": {
                "type": "object",
                "properties": {
                    "resume_text": {
                        "type": "string",
                        "description": "The full resume text.",
                    },
                    "required_keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The required skills/keywords from the job description.",
                    },
                },
                "required": ["resume_text", "required_keywords"],
            },
        },
    }
]

# Map tool names -> the Python functions that run them.
TOOL_FUNCS = {"find_missing_keywords": find_missing_keywords}


# --- The agent loop ----------------------------------------------------------
def run_agent(user_message: str, max_steps: int = 5) -> str:
    """Run the tool-use loop until the model returns a final answer."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a resume-fit assistant. When asked how a resume matches "
                "a role, call the find_missing_keywords tool to check the required "
                "skills, then give a short, honest summary of what's covered and "
                "what's missing. Never invent skills."
            ),
        },
        {"role": "user", "content": user_message},
    ]

    for _ in range(max_steps):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )
        msg = response.choices[0].message

        # No tool call -> the model is done; return its final answer.
        if not msg.tool_calls:
            return msg.content or ""

        # Record the assistant's tool-call turn.
        # NOTE: omit "content" entirely when there are tool calls -- Gemini
        # rejects empty text parts (an empty-string content causes a 400).
        assistant_msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        }
        if msg.content:
            assistant_msg["content"] = msg.content
        messages.append(assistant_msg)

        # Execute each requested tool and feed the results back to the model.
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            print(f"  [agent] calling tool: {name}({list(args.keys())})")
            result = TOOL_FUNCS[name](**args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )

    return "Stopped: reached max steps without a final answer."


# --- Try it ------------------------------------------------------------------
if __name__ == "__main__":
    sample_resume = (
        "Software engineer. Built full-stack apps with Python, Flask, React, "
        "and PostgreSQL. Wrote automated tests with pytest and Selenium. "
        "Comfortable with Git and REST APIs."
    )
    sample_required = ["Python", "React", "Docker", "AWS", "pytest", "Kubernetes"]

    task = (
        "Here is my resume:\n\n"
        f"{sample_resume}\n\n"
        f"The role requires these skills: {', '.join(sample_required)}.\n"
        "Which required skills am I missing, and how is my overall fit?"
    )

    print("Running agent...\n")
    try:
        answer = run_agent(task)
        print("\n--- Agent answer ---")
        print(answer)
    except Exception as e:
        # Print the server's actual error body so failures are diagnosable
        # instead of showing a bare status code.
        body = getattr(getattr(e, "response", None), "text", None)
        print("\nERROR:", repr(e))
        if body:
            print("SERVER SAID:", body)