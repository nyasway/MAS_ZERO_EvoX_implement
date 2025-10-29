import evoagentx.workflow.operators as operator
import examples.aflow.mbpp.optimized.round_7.prompt as prompt_custom
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
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)
        self.test = operator.Test(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the workflow
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        # Generate multiple solutions using CustomCodeGenerate
        solutions = []
        for _ in range(3):  # Generate 3 solutions for diversity
            try:
                solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt_custom.GENERATE_PYTHON_CODE_PROMPT)
                solutions.append(solution['response'])
            except Exception as e:
                return f"Error in generating solutions: {str(e)}"

        # Use ScEnsemble to select the best solution
        try:
            selected_solution = await self.sc_ensemble(solutions=solutions, problem=problem)
        except Exception as e:
            return f"Error in selecting solution: {str(e)}"

        # Test the selected solution
        test_result = await self.test(problem=problem, solution=selected_solution['response'], entry_point=entry_point, benchmark=self.benchmark)
        
        # Only return the solution if it passes the test
        if test_result['result']:
            return test_result['solution']
        else:
            return "The generated solution did not pass the tests."
