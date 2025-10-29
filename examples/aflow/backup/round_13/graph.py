import evoagentx.workflow.operators as operator
import examples.aflow.humaneval.optimized.round_13.prompt as prompt_custom
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
        self.test = operator.Test(self.llm)  # Initialize Test operator
        self.sc_ensemble = operator.ScEnsemble(self.llm)  # Initialize ScEnsemble operator

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the workflow
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt_custom.GENERATE_PYTHON_CODE_PROMPT)

        # Introduce a review step to improve solution quality
        review = await self.custom(input=solution['response'], instruction=prompt_custom.REVIEW_CODE_PROMPT)
        
        # Proceed to testing only if the review is positive
        if "no issues" in review['response'].lower():
            test_result = await self.test(problem=problem, solution=solution['response'], entry_point=entry_point, benchmark=self.benchmark)
            
            if not test_result['result']:
                # If the initial solution fails, use self-consistency ensemble to find the best solution
                solutions = [solution['response']]  # Start with the first solution
                solutions.append(await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt_custom.GENERATE_ALTERNATE_CODE_PROMPT)['response'])
                ensemble_result = self.sc_ensemble(solutions=solutions, problem=problem)
                return ensemble_result['response']

            return test_result['solution']
        else:
            # Handle cases where the review indicates issues
            return "Review indicated issues with the solution; please revise."
