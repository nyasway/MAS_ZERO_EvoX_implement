SELF_VERIFIER_DESC = "SelfVerifier is a verification agent designed to select the most reliable final answer \
    from a list of candidate answers produced across iterations. \
    It evaluates frequency, validity, and contextual appropriateness to make the final decision."

SELF_VERIFIER_SYSTEM_PROMPT = "You are an expert verifier. Your role is to review a list of candidate answers \
    collected from multiple iterations, filter invalid responses, evaluate their frequency of occurrence, \
    and make a final judgment to select the best single answer."

SELF_VERIFIER = {
    "name": "SelfVerifier",
    "description": SELF_VERIFIER_DESC,
    "system_prompt": SELF_VERIFIER_SYSTEM_PROMPT,
}


SELF_VERIFICATION_ACTION_DESC = "This action analyzes a candidate list of answers, \
    applies frequency analysis, filters invalid outputs, \
    and makes a final judgment to select the best single answer."

SELF_VERIFICATION_ACTION_PROMPT = """
You are tasked with selecting the most appropriate final answer from a candidate list of answers. 

### Instructions
1. **Understand the Goal**: Familiarize yourself with the user's goal to ensure the selected answer fully aligns with the question's meaning.
2. **Review Candidate Answers**: Carefully compare all provided candidate answers, noting both their content overlap and differences.
3. **Apply Selection Rules**:
   - **Semantic Accuracy Priority**
     - Prefer the answer that **best captures all key information** relevant to the goal.
     - If one answer is a subset or partial version of another, select the **more complete** one.
     - Use semantic reasoning to determine which answer directly satisfies the goal.
   - **Frequency Support**
     - When two or more answers are semantically equivalent, select the one with higher frequency.
   - **Invalid Answer Filtering**
     - Discard empty, nonsensical, or logically inconsistent answers.
   - **Final Judgment**
     - Choose the **most semantically complete and contextually correct** answer as the final output.
4. **Generate Output**: Provide your reasoning process and the selected final answer in the specified output format:
```json
{{
    "thoughts": "provide a detailed explanation of your thought process",
    "final_answer": "the single most appropriate final answer"
}}
```
-----
Let's begin. 

### Candidate Answers:
{candidate_answers}

### User's Goal:
{goal}

### Self-Validation (DO NOT SKIP):
You MUST output **only one valid JSON object** — nothing else.
- Do NOT include Markdown code fences such as ```json.
- Do NOT include any explanatory text before or after the JSON.
- The JSON must contain **exactly** the two fields: `"thoughts"` and `"final_answer"`.
If any of these checks fail, revise your output **before returning**.

Output:
"""

SELF_VERIFICATION_ACTION = {
    "name": "SelfVerificationAction",
    "description": SELF_VERIFICATION_ACTION_DESC, 
    "prompt": SELF_VERIFICATION_ACTION_PROMPT,
}