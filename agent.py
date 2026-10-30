"""
JD Fit Agent -- Milestone 3 (refactored for robustness)

Key design lesson: tools read large data (resume, JD) from shared MEMORY
instead of making the model pass it through tool-call arguments. The model
only passes SMALL things (a short skills list). This avoids the model having
to re-emit a whole resume/JD as JSON -- which breaks on long text.

Memory (agent_memory.json) holds:
  - resume       : your resume text
  - current_jd   : the job description being assessed

Tool chain the agent runs:
  1. get_resume              -> read the stored resume (so it can summarize)
  2. extract_required_skills -> read the stored JD, return its required skills
  3. find_missing_keywords   -> compare those skills against the stored resume
  4. summarize the fit

Runtime model: Groq (free tier), via its OpenAI-compatible endpoint.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Model configuration -----------------------------------------------------
BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "openai/gpt-oss-120b"
API_KEY = os.environ.get("GROQ_API_KEY")

if not API_KEY:
    raise SystemExit("Set GROQ_API_KEY in your .env file (copy .env.example).")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# --- Persistent memory -------------------------------------------------------
MEMORY_FILE = "agent_memory.json"


def load_memory() -> dict:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_memory(data: dict) -> None:
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


DEFAULT_RESUME = (
    "Software engineer. Built full-stack apps with Python, Flask, React, "
    "and PostgreSQL. Wrote automated tests with pytest and Selenium. "
    "Comfortable with Git and REST APIs."
)


def ensure_resume_in_memory() -> None:
    mem = load_memory()
    if "resume" not in mem:
        mem["resume"] = DEFAULT_RESUME
        save_memory(mem)
        print(f"(Seeded a resume into {MEMORY_FILE} -- edit that file to use your real one.)\n")


RESUME_FILE = "my_resume.txt"


def load_resume_from_file() -> None:
    """If my_resume.txt exists, load its text as the resume in memory."""
    if os.path.exists(RESUME_FILE):
        with open(RESUME_FILE, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            mem = load_memory()
            mem["resume"] = text
            save_memory(mem)
            print(f"(Loaded your resume from {RESUME_FILE} into memory.)\n")


def set_current_jd(jd_text: str) -> None:
    """Store the JD being assessed in memory so tools can read it."""
    mem = load_memory()
    mem["current_jd"] = jd_text
    save_memory(mem)


# --- Tools (they READ from memory; the model passes only small data) ---------
SKILL_VOCAB = [
    "Python", "Java", "C#", "C++", "JavaScript", "TypeScript", "SQL", "Go", "Bash",
    "React", "Angular", "Next.js", "Node.js", "Flask", "Django", "FastAPI", "Spring Boot",
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "SQLite",
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "CI/CD", "Terraform", "Linux",
    "REST", "GraphQL", "gRPC", "WebSockets", "Microservices",
    "pytest", "Selenium", "JUnit", "Jest", "Git", "Agile",
]


def get_resume() -> dict:
    """Retrieve the user's resume from memory (so the model can read it)."""
    mem = load_memory()
    return {"resume": mem.get("resume", "")}


def extract_required_skills() -> dict:
    """Read the stored JD from memory and return the known skills it mentions."""
    mem = load_memory()
    text = mem.get("current_jd", "").lower()
    found = [skill for skill in SKILL_VOCAB if skill.lower() in text]
    return {"required_skills": found}


def find_missing_keywords(required_skills: list) -> dict:
    """Compare the given required skills against the resume stored in memory."""
    mem = load_memory()
    text = mem.get("resume", "").lower()
    present, missing = [], []
    for skill in required_skills:
        (present if skill.lower() in text else missing).append(skill)
    return {"present": present, "missing": missing}


# --- Tool schemas ------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_resume",
            "description": "Retrieve the user's stored resume from memory.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_required_skills",
            "description": "Extract the required technical skills from the stored job description.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_missing_keywords",
            "description": "Given a list of required skills, check which are present or missing in the stored resume.",
            "parameters": {
                "type": "object",
                "properties": {
                    "required_skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The required skills (e.g., from extract_required_skills).",
                    },
                },
                "required": ["required_skills"],
            },
        },
    },
]

TOOL_FUNCS = {
    "get_resume": get_resume,
    "extract_required_skills": extract_required_skills,
    "find_missing_keywords": find_missing_keywords,
}


# --- The agent loop ----------------------------------------------------------
def run_agent(user_message: str, max_steps: int = 8) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a resume-fit assistant. The resume and job description are "
                "stored in memory. You have three tools: get_resume (read the stored "
                "resume), extract_required_skills (get the job's required skills from "
                "the stored JD), and find_missing_keywords (given a skills list, check "
                "which the resume covers). Steps: FIRST call get_resume, THEN call "
                "extract_required_skills, THEN call find_missing_keywords with the "
                "skills from the previous step, then give a short, honest fit summary. "
                "Never invent skills."
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

        if not msg.tool_calls:
            return msg.content or ""

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

        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
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
    load_resume_from_file()    # if my_resume.txt exists, use your real resume
    ensure_resume_in_memory()  # otherwise seed a sample on first run

    # Paste a REAL job description into jd.txt to test it.
    SAMPLE_JD = (
        "We're hiring a Software Engineer to build backend services in Python "
        "and a React frontend. You'll deploy on AWS using Docker and maintain "
        "our CI/CD pipelines. PostgreSQL and REST APIs required; Kubernetes a plus."
    )
    JD_FILE = "jd.txt"
    if os.path.exists(JD_FILE):
        with open(JD_FILE, "r", encoding="utf-8") as f:
            jd_text = f.read().strip()
        print(f"(Using job description from {JD_FILE}.)\n")
    else:
        jd_text = SAMPLE_JD
        print(f"(No {JD_FILE} found -- using the built-in sample. Create {JD_FILE} to test a real JD.)\n")

    set_current_jd(jd_text)  # store the JD so tools can read it (no big arg through the model)

    task = (
        "A resume and a job description are stored in memory. "
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