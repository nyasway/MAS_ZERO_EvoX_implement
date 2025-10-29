import evoagentx.workflow.operators as operator
import examples.aflow.hotpotqa.optimized.round_17.prompt as prompt_custom
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
        self.review_solutions = operator.Custom(self.llm)  # Initialize Custom operator for reviews

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        solutions = []
        for _ in range(3):  # Generate multiple solutions
            solution = await self.answer_generate(input=problem)
            solutions.append(solution['answer'])

        # Review the generated solutions
        reviewed_solutions = await self.review_solutions(input="\n".join(solutions), instruction=prompt_custom.REVIEW_PROMPT)

        # Use QAScEnsemble to select the most frequent solution from reviewed solutions
        final_solution = await self.sc_ensemble(solutions=reviewed_solutions['response'].split('\n'))
        return final_solution['response']
