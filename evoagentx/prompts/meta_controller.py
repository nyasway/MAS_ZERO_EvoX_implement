META_CONTROLLER_DESC = "MetaAgent is a reflective evaluation and coordination agent designed to oversee MAS iterations. \
    Its role is to review sub-task and agent execution results, assess solvability and correctness, \
    and provide structured feedback to improve both sub-task decomposition and agent assignment in the next iteration."

META_CONTROLLER_SYSTEM_PROMPT = "You are an expert meta-agent responsible for monitoring the execution of sub-tasks by multiple agents. \
    You will review input-output pairs of sub-tasks and agents, determine solvability, correctness, \
    and produce structured feedback for improving sub-task decomposition and agent assignment."

META_CONTROLLER = {
    "name": "MetaController",
    "description": META_CONTROLLER_DESC,
    "system_prompt": META_CONTROLLER_SYSTEM_PROMPT,
}


META_FEEDBACK_ACTION_DESC = "This action reviews the performance of agents on assigned sub-tasks, \
    evaluates solvability and correctness, and generates improvement suggestions for both task decomposition and agent assignment."

META_FEEDBACK_ACTION_PROMPT = """
You are tasked with generating feedback for the performance of agents executing sub-tasks in a workflow. Analyze the provided input-output pairs for both sub-tasks and agents, and generate feedback on whether each sub-task was solved correctly and completely.

### Instructions
1. **Understand the workflow**: Familiar yourself with the overall workflow, including its goal, to understand the relationship between the sub-tasks and how outputs from one may serve as inputs for another. 
2. **Analyse the Sub-Task**: Review the sub-task details in the below format to fully understand its objective, requirements, and expected inputs and outputs.
```json
{{
    "name": "subtask_name",
    "description": "A clear and concise explanation of the goal of this sub-task.",
    "reason": "Why this sub-task is necessary and how it contributes to achieving user's goal.",
    "inputs": [
        {{
            "name": "the input's name", 
            "type": "string/int/float/other_type",
            "required": true/false (`false` means the input is the feedback from later sub-task, or the previous output for the current sub-task), 
            "description": "Description of the input's purpose and usage."
        }},
        ...
    ], 
    "outputs": [
        {{
            "name": "the output's name", 
            "type": "string/int/float/other_type",
            "required": true (the `required` field of outputs are always true), 
            "description": "Description of the output produced by this sub-task."
        }},
        ...
    ]
}}
```
3. **Analyse Agent-Level Input-Output Pairs**: Review the provided input-output pairs for agents in the subtask in the "### Agent_Level_Pairs" section, and understand the 'feedback' content in the 'outputs' field to evaluate their performance using the criteria outlined below.
```json
{{
    "agent": "agent_name",
    "inputs": [
        {{
            "name": "the input's name", 
            "type": "string/int/float/other_type",
            "description": "Description of the input's purpose and usage."
        }},
        ...
    ], 
    "outputs": [
        {{
            "name": "the output's name", 
            "type": "string/int/float/other_type",
            "description": "Description of the output produced by this sub-task.",
            "feedback": "If the task is too hard for the agent to solve, set this field to the string 'TOO HARD' to indicate the sub-task is unsolvable in its current form, and list the missing conditions in the description."
        }},
        ...
    ]
}}
```
4. **Analyse Sub-Task Level Input-Output Pairs**: Review the provided input-output pairs for sub-tasks in the "### Subtask_Level_Pairs" section, and evaluate their performance using the criteria outlined below.
```json
{{
    "task": "subtask_name",
    "inputs": [
        {{
            "name": "the input's name", 
            "type": "string/int/float/other_type",
            "description": "Description of the input's purpose and usage."
        }},
        ...
    ], 
    "outputs": [
        {{
            "name": "the output's name", 
            "type": "string/int/float/other_type",
            "description": "Description of the output produced by this sub-task."
        }},
        ...
    ]
}}
``` 
5. **Evaluate Solvability and Correctness Criteria**: For each sub-task and its assigned agents, determine if the execution was valid and solvable based on the following criteria: 
    - Explicit Refusal Signal ('TOO HARD')
      - If an agent outputs the token 'TOO HARD', mark the sub-task as unsolvable in its current form.
      - Suggest further decomposition of the sub-task or reallocation to a different sub-MAS.
    - Input-Output Coherence
      - Based on the input-output pairs.
      - If an output is empty, illogical, or cannot serve as valid input for the next step, mark the sub-task as unsolvable or the agent as malfunctioning.
      - Suggest reconfiguration of the agent or redesign of the sub-task.
    - Independent Completeness
      - Each sub-task must be independently solvable with a complete answer.
      - If the output is partial, missing key details, or only provides incomplete reasoning, consider the sub-task unsolved even if 'TOO HARD' was not explicitly emitted.
      - Suggest further decomposition of the sub-task.
    - Decomposition Completeness
      - Ensure that all agents answers collectively cover the entire scope of the sub-task.
      - If any part of the sub-task is not addressed by the agents, mark it as unsolved and suggest further decomposition.
6. **Generate Structured Feedback**: After evaluating all sub-tasks and agents, generate structured feedback in the specified JSON format to help improve future iterations of task decomposition and agent assignment.
Generated feedback MUST be defined in the following JSON format:
```json
{{
     "task": "subtask_name",
     "agents": [
       {{
         "agent": "agent_name1",
         "feedback": "Analysis of this agent's solvability and correctness."
       }},
       {{
         "agent": "agent_name2",
         "feedback": "..."
       }},
       ...
     ],
     "task_feedback": "Overall analysis of the sub-task: whether it is solvable, complete, and suggestions for decomposition."
}}
```
-----
Let's begin. 

### Subtask_Level_Pairs (input-output pairs of sub-tasks and their results):
{subtask_level_pairs}

### Agent_Level_Pairs (input-output pairs of agents and their results):
{agent_level_pairs}

### User's Goal:
{goal}

### Workflow:
{workflow}

### Self-Validation (DO NOT SKIP)
Before returning your final output:
1. Verify that you have analyzed **all** provided sub-tasks and agents.
2. Ensure that each feedback entry includes **solvability**, **completeness**, and **specific improvement suggestions**.
3. You MUST output **only one single valid JSON object**, with **no Markdown formatting**, **no code fences**, and **no additional text or commentary**. 
If any of these checks fail, revise your output **before returning**.

Output:
"""

META_FEEDBACK_ACTION = {
    "name": "MetaFeedbackAction",
    "description": META_FEEDBACK_ACTION_DESC,
    "prompt": META_FEEDBACK_ACTION_PROMPT,
}