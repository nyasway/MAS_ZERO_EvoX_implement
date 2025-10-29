TASK_PLANNER_DESC = "TaskPlanner is an intelligent task planning agent designed to assist users in achieving their goals. \
    It specializes in breaking down complex tasks into clear, manageable sub-tasks and organizing them in the most efficient sequence." 

TASK_PLANNER_SYSTEM_PROMPT = "You are a highly skilled task planning expert. Your role is to analyze the user's goals, deconstruct complex tasks into actionable and manageable sub-tasks, and organize them in an optimal execution sequence."

TASK_PLANNER = {
    "name": "TaskPlanner", 
    "description": TASK_PLANNER_DESC,
    "system_prompt": TASK_PLANNER_SYSTEM_PROMPT,
}


TASK_PLANNING_ACTION_DESC = "This action analyzes a given task, breaks it down into manageable sub-tasks, and organizes them in the optimal order to help achieve the user's goal efficiently."

TASK_PLANNING_ACTION_PROMPT_OLD = """
Given a user's goal, analyze the task and create a workflow by breaking it down into actionable sub-tasks. Organize these sub-tasks in an optimal execution order for smooth and efficient completion.

### Instructions:
1. **Understand the Goal**: Identify the core objectives and outcomes the user wants to achieve. 
2. **Review the History**: Assess any previously generated task plan to identify gaps or areas needing refinement. 
3. **Consider Suggestions**: Consider user-provided suggestions to improve or optimize the workflow. 

4. **Define Sub-Tasks**: Break the task into logical, actionable sub-tasks based on the complexity of the goal.

4.1 **Principle for Breaking Task**:
- **Simplicity**: Each sub-task is designed to achieve a specific, clearly defined objective. Avoid overloading sub-tasks with multiple objectives. 
- **Modularity**: Ensure that each sub-task is self-contained, reusable, and contributes meaningfully to the overall solution. 
- **Consistency**: Sub-tasks must logically support the user's goal and maintain coherence across the workflow.
- **Optimize Complexity**: Adjust the number of sub-tasks according to task complexity. Highly complex tasks may require more detailed steps, while simpler tasks should remain concise.
- **Avoid Redundancy**: Ensure that there are no overlapping or unnecessary sub-tasks. 

4.2 **Sub-Task Format**: 
Each sub-task should follow the structure below:
```json
{{
    "name": "subtask_name",
    "description": "A clear and concise explanation of the goal of this sub-task.",
    "reason": "Why this sub-task is necessary and how it contributes to achieving user's goal.",
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

### Special Instructions for Programming Tasks
- **Environment Setup and Deployment**: For programming-related tasks, **do not** include sub-tasks related to setting up environments or deployment unless explicitly requested.
- Focus on tasks such as requirements analysis, design, coding, debugging, and testing, etc. 
- Ensure that sub-tasks reflect standard coding practices like module design, coding, and testing.


### Notes:
- Provide clear and concise names for the sub-tasks, inputs, and outputs. 
- Maintain consistency in the flow of inputs and outputs between sub-tasks to ensure seamless integration. 
- The inputs of a sub-task can ONLY be chosen from the user's `goal` and any outputs from its preceding sub-tasks. 
- The inputs of a su-btask should contain SUFFICIENT information to effectivelly address the current sub-task.
- The inputs of a sub-task MUST include the user's input `goal`. 
- The first sub-task must have only one `input_name` "goal" with the following structure:
```json
"inputs": [
    {{
        "name": "goal",
        "type": "string",
        "description": "The user's goal in textual format."
    }}
]
```

### Output Format
Your final output should ALWAYS in the following format:

## Thought 
Provide a brief explanation of your reasoning for breaking down the task and the chosen task structure.  

## Goal
Restate the user's goal clearly and concisely.

## Plan
You MUST provide the workflow plan with detailed sub-tasks in the following JSON format. The description of each sub-task MUST STRICTLY follow the JSON format described in the **Sub-Task Format** section. If a sub-task doesn't require inputs or do not have ouputs, still include `inputs` and `outputs` in the definiton by setting them as empty list. 
```json
{{
    "sub_tasks": [
        {{
            "name": "subtask_name", 
            ...
        }}, 
        {{
            "name": "another_subtask_name", 
            ...
        }},
        ...
    ]
}}
```

-----
Let's begin. 

### History (previously generated task plan):
{history}

### Suggestions (idea of how to design the workflow or suggestions to refine the history plan):
{suggestion}

### User's Goal:
{goal}

Output:
"""

TASK_PLANNING_ACTION_INST = """
Your Task: Given a user's goal, break it down into clear, manageable sub-tasks that are easy to follow and efficient to execute. 

### Instructions:
1. **Understand the Goal**: Identify the core objective the user is trying to achieve. 
2. **Review the History**: Assess any previously generated task plan to identify gaps or areas needing refinement. 
3. **Consider Suggestions**: Use the provided suggestions to understand what should be improved conceptually, but do NOT reuse, edit, or extend the previous workflow.
3.1 **Principle for designing the workflow**:
- You must generate a **completely new workflow** from scratch. 
- Treat the previous workflow as invalid — use it only for contextual awareness. Do NOT continue, extend, or modify any part of it.
- Do NOT reuse any agent in the historical plan as existing agents, design them anew even if they have the same functionality.
- If you don't follow this principle, your output will be **rejected**.

4. **Define Sub-Tasks**: Break the task into logical, actionable sub-tasks based on the complexity of the goal. 

4.1 **Principle for Breaking Task**:
- **Simplicity**: Each sub-task is designed to achieve a specific, clearly defined objective. Avoid overloading sub-tasks with multiple objectives. 
- **Modularity**: Ensure that each sub-task is self-contained, reusable, and contributes meaningfully to the overall solution. 
- **Consistency**: Sub-tasks must logically support the user's goal and maintain coherence across the workflow.
- **Optimize Complexity**: Adjust the number of sub-tasks according to task complexity. Highly complex tasks may require more detailed steps, while simpler tasks should remain concise.
- **Avoid Redundancy**: Ensure that there are no overlapping or unnecessary sub-tasks. 
- **Consider Cycles**: Identify tasks that require iteration or feedback loops, and structure dependencies (by specifying inputs and outputs) accordingly. 

4.2 **Sub-Task Format**: 
Each sub-task should follow the structure below:
```json
{{
    "name": "subtask_name",
    "description": "A clear and concise explanation of the goal of this sub-task.",
    "reason": "Why this sub-task is necessary and how it contributes to achieving user's goal.",
    "inputs": [
        {{
            "name": "the input's name", 
            "type": "string/int/float/other_type",
            "required": true/false (only set to `false` when this input is the feedback from later sub-task, or the previous generated output for the current sub-task),
            "description": "Description of the input's purpose and usage."
        }},
        ...
    ], 
    "outputs": [
        {{
            "name": "the output's name", 
            "type": "string/int/float/other_type",
            "required": true (always set the `required` field of outputs as true), 
            "description": "Description of the output produced by this sub-task."
        }},
        ...
    ]
}}
```

### Special Instructions for Programming Tasks
- **Environment Setup and Deployment**: For programming-related tasks, **do not** include sub-tasks related to setting up environments or deployment unless explicitly requested.
- **Complete Code Generation**: For programming-related tasks, ensure that the final sub-task outputs a complete and working solution.
- **IMPORTANT - Include Full Requirements**: For EVERY code generation tasks, in addition to the outputs from previous sub-tasks, the overall goal (and analysed requirements if any) MUST be included as inputs. This ensures each code generation step maintains full context of what's being built, even when split across multiple steps.

### Notes:
- Provide clear and concise names for the sub-tasks, inputs, and outputs. 
- Maintain consistency in the flow of inputs and outputs between sub-tasks to ensure seamless integration. 
- The inputs of a sub-task can ONLY be chosen from the user's `goal` and any outputs from its preceding sub-tasks. 
- The inputs of a su-btask should contain SUFFICIENT information to effectivelly address the current sub-task.
- The inputs of a sub-task MUST include the user's input `goal`. 
- The first sub-task must have only one `input_name` "goal" with the following structure:
```json
"inputs": [
    {{
        "name": "goal",
        "type": "string",
        "required": true,
        "description": "The user's goal in textual format."
    }}
]
```
- If a sub-task require feedback from a later sub-task (for feedback or refinement), include the later sub-task's output and the current sub-task's output in the current sub-task's inputs and set `"required": false`. 
"""

TASK_PLANNING_ACTION_DEMOS = """
### Examples: 
Below are some generated workflows that follow the given instructions:

Example 1: 
### User's goal: 
Create a Python function that takes two numbers as input and returns their sum.
### Generated Workflow: 
{{
    "sub_tasks": [
        {{
            "name": "code_generation",
            "description": "Generate a Python function that takes two numbers as input and returns their sum.", 
            "reason": "This sub-task ensures that the function correctly implements the required summation logic.", 
            "inputs": [
                {{
                    "name": "goal",
                    "type": "string",
                    "required": true, 
                    "description": "The user's goal in textual format."
                }}
            ],
            "outputs": [
                {{
                    "name": "function_code", 
                    "type": "string", 
                    "required": true, 
                    "description": "The generated Python function code that takes two numbers and returns their sum."
                }}
            ]
        }}
    ]
}}


Example 2: 
### User's goal: 
Given two strings `s` and `t` of lengths `m` and `n` respectively, return the minimum window substring of `s` such that every character in `t` (including duplicates) is included in the window. If there is no such substring, return the empty string "".
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.
### Generated Workflow:
{{
    "sub_tasks": [
        {{
            "name": "task_parsing", 
            "description": "Analyze the problem statement and extract key requirements.", 
            "reason": "This step ensures that we understand the task and its constraints before proceeding with code implementation.", 
            "inputs": [
                {{
                    "name": "goal",
                    "type": "string", 
                    "required": true, 
                    "description": "The user's goal in textual format."
                }}
            ],
            "outputs": [
                {{
                    "name": "problem_analysis",
                    "type": "string",
                    "required": true,
                    "description": "A clear and detailed analysis of the problem, including input/output format and constraints."
                }}
            ]
        }}, 
        {{
            "name": "code_generation", 
            "description": "Generate a Python function that finds the minimum window substring.", 
            "reason": "This step ensures an initial solution is implemented based on the problem analysis.", 
            "inputs": [
                {{
                    "name": "goal",
                    "type": "string", 
                    "required": true, 
                    "description": "The original full programming task requirements."
                }},
                {{
                    "name": "problem_analysis",
                    "type": "string", 
                    "required": true, 
                    "description": "The clear and detailed analysis of the problem."
                }}
            ],
            "outputs": [
                {{
                    "name": "function_code",
                    "type": "string", 
                    "required": true, 
                    "description": "The generated Python function that finds the minimum window substring."
                }}
            ]
        }}
    ]
}}
"""

TASK_PLANNING_OUTPUT_FORMAT = """
### Output Format
Step 1: Restate the user's goal clearly and concisely and briefly explain your reasoning for breaking down the task and the chosen task structure.
Step 2: Then output only the JSON object defining sub_tasks, as shown below. Do not include markdown or any text after the JSON.
- You MUST provide the workflow plan with detailed sub-tasks in the following JSON format. The description of each sub-task MUST STRICTLY follow the JSON format described in the **Sub-Task Format** section. If a sub-task doesn't require inputs or do not have ouputs, still include `inputs` and `outputs` in the definiton by setting them as empty list. 
- If the workflow plan is not a well-formed JSON, your output will be **rejected**.
```json
{{
    "sub_tasks": [
        {{
            "name": "subtask_name", 
            ...
        }}, 
        {{
            "name": "another_subtask_name", 
            ...
        }},
        ...
    ]
}}
```

-----
Let's begin. 

### History (previously generated task plan, for understanding only):
{history}

### Suggestions (idea of how to design the workflow or suggestions to refine the history plan, for consideration only):
{suggestion}

### Mandatory Compliance Rules (DO NOT IGNORE)
You MUST carefully read and strictly implement every improvement or correction mentioned in the Suggestions section.
For each suggestion, explicitly address:
1. What the issue was.
2. What you changed to fix it.
3. How the fix is reflected in this iteration's output.
If any suggestion is ignored, your output will be **rejected**.

### User's Goal:
{goal}

### Self-Validation (DO NOT SKIP)
Before you output your final JSON, perform the following checks:
1. The JSON starts with 'sub_tasks' key.
2. Every sub-task contains all required fields: `name`, `description`, `reason`, `inputs`, and `outputs`.  
3. Each input/output entry includes **name**, **type**, **required**, and **description** fields.  
4. The output is a **valid JSON** object that can be parsed without errors.  
5. No markdown syntax, explanation, or commentary — only raw JSON should be returned.  
6. Every suggestion is addressed. No previous errors are repeated.
If any of these checks fail, revise your output **before returning**.

Output:
"""

TASK_PLANNING_ACTION_PROMPT = TASK_PLANNING_ACTION_INST + TASK_PLANNING_ACTION_DEMOS + TASK_PLANNING_OUTPUT_FORMAT

TASK_PLANNING_ACTION = {
    "name": "TaskPlanning", 
    "description": TASK_PLANNING_ACTION_DESC, 
    "prompt": TASK_PLANNING_ACTION_PROMPT, 
}
