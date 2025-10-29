import evoagentx.workflow.operators as operator
import examples.aflow.humaneval.optimized.round_10.prompt as prompt_custom
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
        # Await generation of the solution
        solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt_custom.GENERATE_PYTHON_CODE_PROMPT)
        
        # Validate the solution using the Test operator
        test_result = await self.test(problem=problem, solution=solution['response'], entry_point=entry_point, benchmark=self.benchmark)
        
        if test_result['result']:
            solutions = [solution['response']]
            # Perform ensemble to refine solution selection
            refined_solution = await self.sc_ensemble(solutions=solutions, problem=problem)
            return refined_solution['response']
        else:
            # Generate feedback to refine the solution
            feedback = await self.custom(input=problem + f"\nFailed Solution: {solution['response']}", instruction=prompt_custom.GENERATE_FEEDBACK_PROMPT)
            revised_solution = await self.custom_code_generate(problem=problem + f"\nFeedback: {feedback['response']}", entry_point=entry_point, instruction=prompt_custom.GENERATE_PYTHON_CODE_PROMPT)
            return revised_solution['response']
