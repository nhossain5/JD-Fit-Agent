"""
JD Fit Agent -- Milestone 2
An agent that CHAINS two tools: it extracts the required skills from a raw
job description, then checks which of those a resume covers.
 
The loop is the same as milestone 1; we just gave the agent a second tool.
  1. The model gets a resume + a raw JD.
  2. It calls extract_required_skills on the JD.
  3. It calls find_missing_keywords with the resume + the extracted skills.
  4. It summarizes the fit.
 
Runtime model: Groq (free tier), via its OpenAI-compatible endpoint.
Swap providers by changing BASE_URL, MODEL, and the key -- nothing else.
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

# --- Tool 1: extract required skills from a raw JD ---------------------------
# A simple known-skills list the extractor scans for. Deterministic on purpose.
SKILL_VOCAB = [
    "Python", "Java", "C#", "C++", "JavaScript", "TypeScript", "SQL", "Go", "Bash",
    "React", "Angular", "Next.js", "Node.js", "Flask", "Django", "FastAPI", "Spring Boot",
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "SQLite",
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "CI/CD", "Terraform", "Linux",
    "REST", "GraphQL", "gRPC", "WebSockets", "Microservices",
    "pytest", "Selenium", "JUnit", "Jest", "Git", "Agile",
]

def extract_required_skills(jd_text: str) -> dict:
    """Scan a raw job description for known skills from SKILL_VOCAB.
 
    A tool stays 'dumb': it just reports which known skills appear in the text.
    The AGENT decides to call this first, then feeds the result into
    find_missing_keywords -- that's the two-tool chain.
    """
    text = jd_text.lower()
    found = [skill for skill in SKILL_VOCAB if skill.lower() in text]
    return {"required_skills": found}

# --- Tool 2: check which required skills the resume covers --------------------
def find_missing_keywords(resume: str, required_skills: list) -> dict:
    """Deterministically check which required skills appear in the resume."""
    text = resume.lower()
    present, missing = [], []
    for skill in required_skills:
        (present if skill.lower() in text else missing).append(skill)
    return {"present": present, "missing": missing}

# --- Tool schemas the model sees (OpenAI function-calling format) -------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "extract_required_skills",
            "description": "Extract the required technical skills mentioned in a raw job description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "jd_text": {
                        "type": "string",
                        "description": "The full, raw job description text.",
                    },
                },
                "required": ["jd_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_missing_keywords",
            "description": "Check which required skills are present or missing in a resume.",
            "parameters": {
                "type": "object",
                "properties": {
                    "resume": {
                        "type": "string",
                        "description": "The full resume text.",
                    },
                    "required_skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The required skills (e.g., from extract_required_skills).",
                    },
                },
                "required": ["resume", "required_skills"],
            },
        },
    },
]

# Map tool names -> the Python functions that run them.
TOOL_FUNCS = {
    "extract_required_skills": extract_required_skills,
    "find_missing_keywords": find_missing_keywords,
}


# --- The agent loop ----------------------------------------------------------
def run_agent(user_message: str, max_steps: int = 6) -> str:
    """Run the tool-use loop until the model returns a final answer."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a resume-fit assistant with two tools: "
                "extract_required_skills (pulls required skills out of a raw job "
                "description) and find_missing_keywords (checks which skills a resume "
                "covers). When given a resume and a raw job description, FIRST call "
                "extract_required_skills on the JD, THEN call find_missing_keywords with "
                "the resume and the extracted skills, then give a short, honest fit "
                "summary. Never invent skills."
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
        # Omit "content" when empty -- some providers reject empty text parts.
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
    sample_jd = (
        "We're hiring a Software Engineer to build backend services in Python "
        "and a React frontend. You'll deploy on AWS using Docker and maintain "
        "our CI/CD pipelines. PostgreSQL and REST APIs required; Kubernetes a plus."
    )
    task = (
        "Here is my resume:\n\n"
        f"{sample_resume}\n\n"
        "Here is the job description:\n\n"
        f"{sample_jd}\n\n"
        "Which required skills am I missing, and how is my overall fit?"
    )
 
    print("Running agent...\n")
    try:
        answer = run_agent(task)
        print("\n--- Agent answer ---")
        print(answer)
    except Exception as e:
        body = getattr(getattr(e, "response", None), "text", None)
        print("\nERROR:", repr(e))
        if body:
            print("SERVER SAID:", body)