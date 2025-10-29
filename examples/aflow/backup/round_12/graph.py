import evoagentx.workflow.operators as operator
import examples.aflow.humaneval.optimized.round_12.prompt as prompt_custom
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
        solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt_custom.GENERATE_PYTHON_CODE_PROMPT)
        
        # New review step added here
        review_response = await self.custom(input=solution['response'], instruction=prompt_custom.REVIEW_CODE_PROMPT)
        
        # If review suggests changes, use that; otherwise, proceed with the generated code
        final_solution = review_response['response'] if review_response['response'] else solution['response']
        
        test_result = await self.test(problem=problem, solution=final_solution, entry_point=entry_point, benchmark=self.benchmark)
        
        if not test_result['result']:
            solutions = [final_solution]
            solutions.append(await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt_custom.GENERATE_ALTERNATE_CODE_PROMPT)['response'])
            ensemble_result = self.sc_ensemble(solutions=solutions, problem=problem)
            return ensemble_result['response']

        return test_result['solution']
