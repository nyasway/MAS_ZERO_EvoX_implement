import evoagentx.workflow.operators as operator
import examples.aflow.humaneval.optimized.round_4.prompt as prompt_custom
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
        # Generate solution using custom code generator
        solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt_custom.GENERATE_PYTHON_CODE_PROMPT)
        
        # Test the generated solution
        test_result = self.test(problem=problem, solution=solution['response'], entry_point=entry_point, benchmark=self.benchmark)
        
        # Verify test result, and if failed, try modifying the solution
        if not test_result['result']:
            # Attempt to modify solution and re-test
            solution = await self.custom_code_generate(problem=f"{problem}\n# Attempt to correct errors", entry_point=entry_point, instruction=prompt_custom.GENERATE_PYTHON_CODE_PROMPT)
            test_result = self.test(problem=problem, solution=solution['response'], entry_point=entry_point, benchmark=self.benchmark)
        
        # Return the solution regardless of test result
        return solution['response'] if test_result['result'] else f"Modified solution failed: {solution['response']}"
