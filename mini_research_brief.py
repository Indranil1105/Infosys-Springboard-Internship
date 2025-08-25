from dotenv import load_dotenv
import os
import json
import re
from typing import List
from pydantic import BaseModel, ValidationError

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

# -----------------------------
# Define Schema
# -----------------------------
class ResearchBrief(BaseModel):
    title: str
    problem_statement: str   # <= 2 sentences
    key_questions: List[str] # 1–3 items
    method_brief: List[str]  # 2–4 items
    deliverables: List[str]  # 2–3 items


# -----------------------------
# Load API Key
# -----------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ Error: Missing Google API key. Please set GOOGLE_API_KEY in your .env file.")
    exit(1)

# -----------------------------
# Initialize Gemini Model
# -----------------------------
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    model_provider="google-genai",
    temperature=0,
    max_retries=2,
    max_tokens=None,
    timeout=None,
    api_key=api_key
)

# -----------------------------
# Define Tools (none for now)
# -----------------------------
TOOLS = []

# -----------------------------
# Prompt Template
# -----------------------------
prompt = """You are a precise research planning assistant.

Given a user-provided research topic, produce a very short research brief.

The output MUST be ONLY a valid JSON object following this schema:
{
  "title": "string",
  "problem_statement": "string (<= 2 sentences)",
  "key_questions": ["1–3 items"],
  "method_brief": ["2–4 items"],
  "deliverables": ["2–3 items"]
}

Constraints:
- Return ONLY JSON (no extra text, no markdown fences).
- Be concise, practical, and actionable.
- Do not exceed limits for each field.
"""

# -----------------------------
# Create Agent
# -----------------------------
agent = create_react_agent(
    model=model,
    tools=TOOLS,
    prompt=prompt,
)

# -----------------------------
# Helper: Extract JSON
# -----------------------------
def extract_json(text: str) -> str:
    """Extract JSON object from text (removes markdown fences if present)."""
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        return match.group(1)
    return text.strip()

# -----------------------------
# Main function
# -----------------------------
def main():
    print("📌 Mini Research Brief Generator")
    topic = input("Enter your research topic: ").strip()

    if not topic:
        print("❌ Error: Empty topic provided. Please enter a valid research topic.")
        return

    try:
        # Ask the agent to create research brief
        result = agent.invoke({"messages": [("user", f"Create a research brief for: {topic}")]}).get("messages", [])
        final_text = ""
        if result:
            last_msg = result[-1]
            final_text = getattr(last_msg, "content", "") or str(last_msg)

        # Extract and parse JSON
        clean_json = extract_json(final_text)
        data = json.loads(clean_json)

        # Validate against schema
        brief = ResearchBrief(**data)

        # Print JSON
        print("\n✅ JSON Output:")
        print(json.dumps(brief.dict(), indent=2))

        # Print Markdown preview
        print("\n📑 Markdown Preview:")
        print(f"# {brief.title}\n")
        print(f"**Problem Statement:** {brief.problem_statement}\n")
        print("### Key Questions")
        for q in brief.key_questions:
            print(f"- {q}")
        print("\n### Method Brief")
        for m in brief.method_brief:
            print(f"- {m}")
        print("\n### Deliverables")
        for d in brief.deliverables:
            print(f"- {d}")

    except json.JSONDecodeError:
        print("❌ Error: Model did not return valid JSON.")
        print("Raw output was:\n", final_text)
    except ValidationError as ve:
        print("❌ Error: JSON did not match schema.")
        print(ve)
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")


    

    # Ensure samples folder exists
    os.makedirs("samples", exist_ok=True)

    # Save output to a JSON file
    file_safe_title = topic.replace(" ", "_").lower()
    output_path = f"samples/{file_safe_title}.json"

    with open(output_path, "w", encoding="utf-8") as f:
         json.dump(brief.dict(), f, indent=2, ensure_ascii=False)

    print(f"\n💾 JSON output saved to {output_path}")

# -----------------------------
# Run Script
# -----------------------------
if __name__ == "__main__":
    main()
