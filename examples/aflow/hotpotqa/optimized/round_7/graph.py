import evoagentx.workflow.operators as operator
import examples.aflow.hotpotqa.optimized.round_7.prompt as prompt_custom
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
        self.sc_ensemble = operator.QAScEnsemble(self.llm)  # Initialize QAScEnsemble
    
    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        solutions = []
        for _ in range(3):  # Generate multiple solutions to enhance self-consistency
            solution = await self.answer_generate(input=problem)
            solutions.append(solution['answer'])
        
        # Use QAScEnsemble to select the most frequent solution
        final_solution = await self.sc_ensemble(solutions=solutions)
        
        # Introduce a feedback loop to verify and improve the final solution
        evaluation = await self.custom(input=problem+f" solution:{final_solution['response']}", instruction=prompt_custom.EVALUATE_PROMPT)
        
        # If the evaluation suggests improvements, refine the solution
        if evaluation['response'] != "Approved":
            improved_solution_options = []
            for _ in range(2):  # Generate a couple of improved solutions
                improved_solution = await self.answer_generate(input=problem+f" feedback:{evaluation['response']}")
                improved_solution_options.append(improved_solution['answer'])

            # Verify the improved solutions using QAScEnsemble
            verified_solution = await self.sc_ensemble(solutions=improved_solution_options)
            
            # Self-review mechanism to further improve the verified solution if necessary
            review = await self.custom(input=problem+f" verified solution:{verified_solution['response']}", instruction=prompt_custom.REVIEW_PROMPT)
            if review['response'] != "Approved":
                final_reviewed_solution = await self.answer_generate(input=problem+f" feedback:{review['response']}")
                return final_reviewed_solution['answer']
            return verified_solution['response']
        
        return final_solution['response']
