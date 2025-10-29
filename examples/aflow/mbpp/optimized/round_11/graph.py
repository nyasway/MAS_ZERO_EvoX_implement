import evoagentx.workflow.operators as operator
import examples.aflow.mbpp.optimized.round_11.prompt as prompt_custom
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
        self.test = operator.Test(self.llm)  # Initialize the Test operator

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the workflow
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        # Generate standard code
        solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt_custom.GENERATE_PYTHON_CODE_PROMPT)
        
        # Test the generated solution
        test_result = await self.test(problem=problem, solution=solution['response'], entry_point=entry_point, benchmark=self.benchmark)
        if test_result['result']:
            return test_result['solution']
        else:
            # If the initial solution fails, attempt to improve it
            improved_solution = await self.custom(input=problem + f" Previous solution failed. Error encountered, please improve: {solution['response']}", instruction=prompt_custom.IMPROVE_PYTHON_CODE_PROMPT)
            return improved_solution['response']
