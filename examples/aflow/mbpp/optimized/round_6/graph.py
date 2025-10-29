import evoagentx.workflow.operators as operator
import examples.aflow.mbpp.optimized.round_6.prompt as prompt_custom
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
        self.code_review = operator.Custom(self.llm)  # Adding a code review operator to enhance quality

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the workflow
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        # Generate multiple solutions using CustomCodeGenerate
        solutions = []
        for _ in range(3):  # Generate 3 solutions for diversity
            solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt_custom.GENERATE_PYTHON_CODE_PROMPT)
            solutions.append(solution['response'])

        # Use ScEnsemble to select the best solution
        selected_solution = await self.sc_ensemble(solutions=solutions, problem=problem)

        # Perform a code review to check for potential improvements or errors
        revised_solution = await self.code_review(input=selected_solution['response'], instruction=prompt_custom.CODE_REVIEW_PROMPT)

        # Test the revised solution
        test_result = await self.test(problem=problem, solution=revised_solution['response'], entry_point=entry_point, benchmark=self.benchmark)
        
        # Only return the solution if it passes the test
        if test_result['result']:
            return test_result['solution']
        else:
            return "The generated solution did not pass the tests."
