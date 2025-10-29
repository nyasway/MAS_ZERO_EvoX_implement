import evoagentx.workflow.operators as operator
import examples.aflow.hotpotqa.optimized.round_4.prompt as prompt_custom
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
        self.answer_generate = operator.AnswerGenerate(self.llm)
        # Add QAScEnsemble operator for self-consistency
        self.sc_ensemble = operator.QAScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate initial solutions
        solutions = []
        for _ in range(3):  # Generate multiple solutions for diversity
            solution = await self.answer_generate(input=problem)
            solutions.append(solution['answer'])
        
        # Use QAScEnsemble to select the best solution
        best_solution = await self.sc_ensemble(solutions=solutions)
        
        # Review and refine the best solution using the custom operator
        reviewed_solution = await self.custom(input=best_solution['response'], instruction=prompt_custom.REVIEW_PROMPT)
        
        return reviewed_solution['response']
