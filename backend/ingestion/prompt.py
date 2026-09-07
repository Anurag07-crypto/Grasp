def PROMPT(context):
    CONCEPT_EXTRACTION_PROMPT = f"""
You are a Kubernetes learning-structure extractor.
Your task is to analyze the provided Kubernetes text chunk and extract:
1. Important Kubernetes concepts explicitly discussed in the chunk.
2. Prerequisites for each concept, only when supported by the text.
Rules:
- Extract meaningful Kubernetes concepts, resources, components,
  mechanisms, commands, or technical topics.
- Do not invent concepts that are not discussed in the chunk.
- Do not assume prerequisites from general Kubernetes knowledge.
- If no prerequisite is supported by the text, return an empty list.
- Avoid duplicate concepts.
- Keep concept names concise and canonical.
- Return ONLY valid JSON.
Output format:
[
  {
    "concept": "Concept name",
    "prerequisites": [
      "Prerequisite concept"
    ]
  }
]
Kubernetes text chunk:
{context}
"""

PROMPT()