SOLVE_MATH_PROBLEM_PROMPT = r"""
Answer the math question. The answer should be in box format, e.g., \boxed{123}

Problem: """

VALIDATE_MATH_SOLUTION_PROMPT = r"""
Ensure the solution is correctly formatted in box format, e.g., \boxed{123}. If not, correct it.

Problem: """

REVISE_MATH_SOLUTION_PROMPT = r"""
Revise the solution based on execution feedback to ensure it is correct and in box format, e.g., \boxed{123}.

Problem: """