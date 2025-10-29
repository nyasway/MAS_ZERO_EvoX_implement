from evoagentx.models import OpenAILLMConfig, OpenAILLM 
from evoagentx.benchmark import MATH
from evoagentx.evaluators import Evaluator 
from evoagentx.core.callbacks import suppress_logger_info
from evoagentx.workflow.operators import Operator, AnswerGenerate, ScEnsemble 
from evoagentx.workflow.action_graph import ActionGraph
from evoagentx.models.model_configs import LLMConfig

import os 
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY_O = os.getenv("OPENAI_API_KEY_O")
OPENAI_API_KEY_E = os.getenv("OPENAI_API_KEY_E")
SILICONFLOW_API_KEY_O = os.getenv("SILICONFLOW_API_KEY_O")
SILICONFLOW_API_KEY_E = os.getenv("SILICONFLOW_API_KEY_E")

class MathSplits(MATH):

    def _load_data(self):
        # load the original test data 
        super()._load_data()
        # split the data into dev and test
        import numpy as np 
        np.random.seed(42)
        permutation = np.random.permutation(len(self._test_data))
        full_test_data = self._test_data
        self._test_data = [full_test_data[idx] for idx in permutation[50:150]]

class MATHActionGraph(ActionGraph):

    def __init__(self, llm_config: LLMConfig, **kwargs):

        name = kwargs.pop("name") if "name" in kwargs else "Simple MATH Workflow"
        description = kwargs.pop("description") if "description" in kwargs else \
            "This is a simple MATH workflow that use self-consistency to make predictions."
        super().__init__(name=name, description=description, llm_config=llm_config, **kwargs)
        self.answer_generate = AnswerGenerate(self._llm)
        self.sc_ensemble = ScEnsemble(self._llm)
        
    def execute(self, problem: str) -> dict:

        solutions = [] 
        for _ in range(3):
            response = self.answer_generate(input=problem)
            answer = response["answer"]
            solutions.append(answer)
        ensemble_result = self.sc_ensemble(solutions=solutions, problem=problem)
        best_answer = ensemble_result["response"]
        return {"answer": best_answer}
    
    async def async_execute(self, problem: str) -> dict:
        solutions = [] 
        for _ in range(3):
            response = await self.answer_generate(input=problem)
            answer = response["answer"]
            solutions.append(answer)
        ensemble_result = await self.sc_ensemble(solutions=solutions, problem=problem)
        best_answer = ensemble_result["response"]
        return {"answer": best_answer}


def main(): 

    llm_config = OpenAILLMConfig(
        model="gpt-4o",
        openai_key= OPENAI_API_KEY_E,
        temperature=0.4,
        max_tokens=1000
        )
    llm = OpenAILLM(config=llm_config)

    benchmark = MathSplits()

    workflow = MATHActionGraph(
        llm_config=llm_config,
        description="This workflow aims to address MATH tasks."
    )

    def collate_func(example: dict) -> dict:
        return {"problem": example["problem"]}


    def output_postprocess_func(output: dict) -> dict:
        """
        Args:
            output (dict): The output from the workflow.

        Returns: 
            The processed output that can be used to compute the metrics. The output will be directly passed to the benchmark's `evaluate` method. 
        """
        return output["answer"]

    evaluator = Evaluator(
        llm=llm, 
        collate_func=collate_func,
        output_postprocess_func=output_postprocess_func,
        verbose=True, 
        num_workers=3 
    )

    with suppress_logger_info():
        results = evaluator.evaluate(
            graph=workflow, 
            benchmark=benchmark, 
            eval_mode="test", # Evaluation split: train / dev / test 
            sample_k=100 # If set, randomly sample k examples from the benchmark for evaluation  
        )
    
    print("Evaluation metrics: ", results)

if __name__ == "__main__":
    main()