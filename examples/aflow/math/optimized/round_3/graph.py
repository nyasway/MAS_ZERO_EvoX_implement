import evoagentx.workflow.operators as operator
import examples.aflow.math.optimized.round_3.prompt as prompt_custom
from evoagentx.models.model_configs import LLMConfig
from evoagentx.benchmark.benchmark import Benchmark
from evoagentx.models.model_utils import create_llm_instance

class Workflow:

    def __init__(
        self,
        name: str,
        llm_config: LLMConfig,
        benchmark: Benchmark
    ):
        self.name = name
        self.llm = create_llm_instance(llm_config)
        self.benchmark = benchmark
        self.custom = operator.Custom(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
    
    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        solutions = []
        for _ in range(3):  # Generate multiple solutions
            response = await self.custom(input=problem, instruction=prompt_custom.SOLVE_MATH_PROBLEM_PROMPT)
            solutions.append(response['response'])

        # Use ScEnsemble to select the most consistent solution
        solution = await self.sc_ensemble(solutions=solutions, problem=problem)
        
        # Validate the solution to ensure it is in the correct format
        if '\\boxed{' not in solution['response']:
            solution['response'] = await self.validate_solution(solution['response'], problem)
        
        return solution['response']
    
    async def validate_solution(self, solution: str, problem: str) -> str:
        """
        Validate and correct the solution format
        """
        response = await self.custom(input=problem+f" solution:{solution}", instruction=prompt_custom.VALIDATE_MATH_SOLUTION_PROMPT)
        return response['response']
