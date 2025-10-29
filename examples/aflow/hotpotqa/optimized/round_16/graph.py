import evoagentx.workflow.operators as operator
import examples.aflow.hotpotqa.optimized.round_16.prompt as prompt_custom
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
        self.answer_generate = operator.AnswerGenerate(self.llm)
        self.sc_ensemble = operator.QAScEnsemble(self.llm)  # Initialize QAScEnsemble
        self.verify_solution = operator.Custom(self.llm)  # Initialize Custom operator for verification

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        solutions = []
        for _ in range(3):  # Generate multiple solutions
            solution = await self.answer_generate(input=problem)
            solutions.append(solution['answer'])

        # Use QAScEnsemble to select the most frequent solution
        final_solution = await self.sc_ensemble(solutions=solutions)

        # Verify the selected solution before returning
        verified_solution = await self.verify_solution(input=final_solution['response'], instruction=prompt_custom.VERIFY_PROMPT)
        
        return verified_solution['response']
