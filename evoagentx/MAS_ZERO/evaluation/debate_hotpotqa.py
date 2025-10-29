import os 
from dotenv import load_dotenv
from tqdm import tqdm

from evoagentx.frameworks.multi_agent_debate.debate import MultiAgentDebateActionGraph
from evoagentx.core.logging import logger
from evoagentx.models import OpenAILLMConfig, OpenAILLM, SiliconFlowConfig, SiliconFlowLLM
from evoagentx.benchmark import benchmark, HotPotQA

load_dotenv()
OPENAI_API_KEY_O = os.getenv("OPENAI_API_KEY_O")
OPENAI_API_KEY_E = os.getenv("OPENAI_API_KEY_E")
SILICONFLOW_API_KEY_O = os.getenv("SILICONFLOW_API_KEY_O")
SILICONFLOW_API_KEY_E = os.getenv("SILICONFLOW_API_KEY_E")

def execute(
        benchmark: benchmark, 
        debate: MultiAgentDebateActionGraph, 
        num_agents: int = 3, 
        num_rounds: int = 3, 
        verbose: bool = True
        ):
    """
    Run MAS-ZERO on a subset of the benchmark test data concurrently and compute average F1.

    Args:
        benchmark: Benchmark dataset instance (e.g., HotPotQA)
        mz: MAS_ZERO system instance
        max_concurrent_tasks: Max number of concurrent evaluations

    Returns:
        (avg_f1, all_failed): Tuple of average F1 score and whether all tasks failed
    """
    benchmark._load_data()
    import numpy as np
    np.random.seed(42)
    permutation = np.random.permutation(len(benchmark._dev_data))
    full_test_data = benchmark._dev_data
    data = [full_test_data[idx] for idx in permutation[50:150]]

    if not data:
        logger.warning("No data to evaluate. Returning zeros.")
        return 0.0, True
    
    results = []
    iterator = tqdm(data, desc=f"Evaluating {benchmark.name} problems", ncols=100) if verbose else data

    for example in iterator:
        question = example["question"]
        paragraphs = [item[1] for item in example["context"] if isinstance(item[1], list)]
        context_str = "\n".join(" ".join(paragraph) for paragraph in paragraphs)
        goal = f"Context: {context_str}\n\nQuestion: {question}\n\nAnswer:"
        if not goal:
                logger.warning("Missing 'question' field in example.")
        
        try:
            prediction = debate.execute(
                problem=goal,
                num_agents=num_agents,
                num_rounds=num_rounds,
                judge_mode="llm_judge",
                return_transcript=True,
                )["final_answer"]
            label = benchmark.get_label(example)
            metrics = benchmark.evaluate(prediction, label)
            f1 = metrics.get("f1", 0.0)
            logger.info(f"Evaluated question: {question[:50]}... | Prediction: {prediction} | Label: {label} | F1: {f1:.3f}")
            results.append(f1)

        except Exception as e:
            logger.warning(f"Failed on evaluating question {question[:50]}... : {e}")
            results.append(None)

    # Handle invalid results
    valid_results = [0.0 if r is None else r for r in results]
    all_failed = all(r is None for r in results)

    if not valid_results:
        logger.warning("No valid results. Returning zeros.")
        avg_metrics = 0.0
    else:
        avg_metrics = sum(valid_results) / len(valid_results)

    logger.info(f"Evaluation complete. Average F1: {avg_metrics:.3f}, All failed: {all_failed}")
    return avg_metrics, all_failed 

    
def main():
    """Main entry point to run MAS-ZERO evaluation on HotPotQA benchmark."""
    logger.info("Initializing models and benchmark...")

    executor_config = OpenAILLMConfig(
        model="gpt-4o-mini",
        openai_key= OPENAI_API_KEY_E,
        temperature=0.4,
        max_tokens=1000
        )
    
    debate = MultiAgentDebateActionGraph(
    name="Simple Debate",
    description="Basic multi-agent debate example",
    llm_config=executor_config,
    )

    benchmark = HotPotQA()

    # Run async evaluation
    logger.info("Starting evaluation process...")
    avg_f1, all_failed = execute(benchmark=benchmark, debate=debate)

    logger.info(f"Final Average F1 Score on HotPotQA: {avg_f1:.3f}")
    logger.info(f"All examples failed? {all_failed}")

if __name__ == "__main__":
    main()