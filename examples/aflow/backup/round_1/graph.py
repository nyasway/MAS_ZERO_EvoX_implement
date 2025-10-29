import evoagentx.workflow.operators as operator
import examples.aflow.humaneval.optimized.round_1.prompt as prompt_custom
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
        self.test = operator.Test(self.llm)  # Added Test operator

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the workflow
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        # Await generation of the solution
        solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt_custom.GENERATE_PYTHON_CODE_PROMPT)
        
        # Validate the solution using the Test operator
        test_result = await self.test(problem=problem, solution=solution['response'], entry_point=entry_point, benchmark=self.benchmark)
        
        if test_result['result']:
            return solution['response']
        else:
            return "Solution failed the test cases."
