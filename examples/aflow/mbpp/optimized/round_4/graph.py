import evoagentx.workflow.operators as operator
import examples.aflow.mbpp.optimized.round_4.prompt as prompt_custom
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
        self.sc_ensemble = operator.ScEnsemble(self.llm)  # Adding ScEnsemble operator
        self.test = operator.Test(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the workflow
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        # Generate multiple solutions using CustomCodeGenerate
        solutions = [await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt_custom.GENERATE_PYTHON_CODE_PROMPT) for _ in range(3)]
        
        # Select the most frequent solution using ScEnsemble
        selected_solution = self.sc_ensemble(solutions=[s['response'] for s in solutions], problem=problem)['response']
        
        # Test the selected solution
        test_result = await self.test(problem=problem, solution=selected_solution, entry_point=entry_point, benchmark=self.benchmark)
        
        # Only return the solution if it passes the test
        if test_result['result']:
            return test_result['solution']
        else:
            return "The generated solution did not pass the tests."
