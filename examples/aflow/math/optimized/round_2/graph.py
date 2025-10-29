import evoagentx.workflow.operators as operator
import examples.aflow.math.optimized.round_2.prompt as prompt_custom
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
        return solution['response']
