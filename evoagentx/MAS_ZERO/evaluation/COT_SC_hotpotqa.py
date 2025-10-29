from evoagentx.models import OpenAILLMConfig, OpenAILLM, SiliconFlowConfig, SiliconFlowLLM
from evoagentx.benchmark import HotPotQA
from evoagentx.workflow import QAActionGraph 
from evoagentx.evaluators import Evaluator 
from evoagentx.core.callbacks import suppress_logger_info

import os 
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY_O = os.getenv("OPENAI_API_KEY_O")
OPENAI_API_KEY_E = os.getenv("OPENAI_API_KEY_E")
SILICONFLOW_API_KEY_O = os.getenv("SILICONFLOW_API_KEY_O")
SILICONFLOW_API_KEY_E = os.getenv("SILICONFLOW_API_KEY_E")

class HotPotQASplits(HotPotQA):

    def _load_data(self):
        # load the original test data 
        super()._load_data()
        # split the data into train, dev and test
        import numpy as np 
        np.random.seed(42)
        permutation = np.random.permutation(len(self._dev_data))
        full_test_data = self._dev_data 
        self._test_data = [full_test_data[idx] for idx in permutation[50:150]]


def main(): 

    llm_config = SiliconFlowConfig(
        model="Pro/deepseek-ai/DeepSeek-V3",
        siliconflow_key=SILICONFLOW_API_KEY_E,
        temperature=0.4,
        max_tokens=1000)
    llm = SiliconFlowLLM(config=llm_config)

    # llm_config = OpenAILLMConfig(
    #     model="gpt-4o",
    #     openai_key= OPENAI_API_KEY_E,
    #     temperature=0.4,
    #     max_tokens=1000
    #     )
    # llm = OpenAILLM(config=llm_config)

    # benchmark = HotPotQA(mode="dev")
    benchmark = HotPotQASplits()

    workflow = QAActionGraph(
        llm_config=llm_config,
        description="This workflow aims to address multi-hop QA tasks."
    )

    def collate_func(example: dict) -> dict:
        """
        Args:
            example (dict): A dictionary containing the raw example data.

        Returns: 
            The expected input for the (custom) workflow.
        """
        problem = "Question: {}\n\n".format(example["question"])
        context_list = []
        for item in example["context"]:
            context = "Title: {}\nText: {}".format(item[0], " ".join([t.strip() for t in item[1]]))
            context_list.append(context)
        context = "\n\n".join(context_list)
        problem += "Context: {}\n\n".format(context)
        problem += "Answer:" 
        return {"problem": problem}


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