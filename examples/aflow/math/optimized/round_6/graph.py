import evoagentx.workflow.operators as operator
import examples.aflow.math.optimized.round_6.prompt as prompt_custom
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
        self.programmer = operator.Programmer(self.llm)
    
    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        solutions = []
        for _ in range(3):
            response = await self.custom(input=problem, instruction=prompt_custom.SOLVE_MATH_PROBLEM_PROMPT)
            solutions.append(response['response'])

        solution = await self.sc_ensemble(solutions=solutions, problem=problem)
        
        if '\\boxed{' not in solution['response']:
            solution['response'] = await self.validate_solution(solution['response'], problem)
        
        execution_result = await self.programmer(problem=problem, analysis=solution['response'])
        
        if execution_result['output'] != solution['response']:
            execution_result['output'] = await self.validate_solution(execution_result['output'], problem)
        
        return execution_result['output']
    
    async def validate_solution(self, solution: str, problem: str) -> str:
        """
        Validate and correct the solution format
        """
        response = await self.custom(input=problem+f" solution:{solution}", instruction=prompt_custom.VALIDATE_MATH_SOLUTION_PROMPT)
        return response['response']
