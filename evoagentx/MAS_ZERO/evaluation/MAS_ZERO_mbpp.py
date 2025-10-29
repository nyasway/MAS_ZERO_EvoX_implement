import os
from dotenv import load_dotenv
from tqdm import tqdm

from evoagentx.core.logging import logger
from evoagentx.models import OpenAILLMConfig, OpenAILLM, SiliconFlowConfig, SiliconFlowLLM
from evoagentx.benchmark import benchmark, HumanEval
from evoagentx.MAS_ZERO.MAS_ZERO import MAS_ZERO

load_dotenv()
OPENAI_API_KEY_O = os.getenv("OPENAI_API_KEY_O")
OPENAI_API_KEY_E = os.getenv("OPENAI_API_KEY_E")
SILICONFLOW_API_KEY_O = os.getenv("SILICONFLOW_API_KEY_O")
SILICONFLOW_API_KEY_E = os.getenv("SILICONFLOW_API_KEY_E")

def execute(benchmark: benchmark, mz: MAS_ZERO, interation_num: int = 5, verbose: bool = True):
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
    np.random.seed(43)
    permutation = np.random.permutation(len(benchmark._test_data))
    full_test_data = benchmark._test_data
    data = [full_test_data[idx] for idx in permutation[50:51]]
    # data = benchmark._test_data

    if not data:
        logger.warning("No data to evaluate. Returning zeros.")
        return 0.0, True
    
    results = []
    iterator = tqdm(data, desc=f"Evaluating {benchmark.name} problems", ncols=100) if verbose else data

    for example in iterator:
        prompt, entry_point = example["prompt"], example["entry_point"]
        instr = "Return only the code implementation (not explanations or test results)"
        goal = f"The problem to be solved is: {prompt}\n\nThe function to be implemented is: {entry_point}\n\n{instr}\n\nImplementation:"
        # goal = f"The prompt of the problem: {prompt}\n\nThe name of the function to be tested: {entry_point}"
        if not goal:
                logger.warning("Missing 'question' field in example.")
        
        try:
            prediction = mz.execute(goal=goal, iteration_num=interation_num)
            label = benchmark.get_label(example)
            metrics = benchmark.evaluate(prediction, label)
            pass_at_1 = metrics.get("pass@1", 0.0)
            logger.info(f"Evaluated question: {goal[:50]}... | Prediction: {prediction} | Label: {label} | Pass@1: {pass_at_1:.3f}")
            results.append(pass_at_1)

        except Exception as e:
            logger.warning(f"Failed on evaluating question {goal[:50]}... : {e}")
            results.append(None)

    # Handle invalid results
    valid_results = [0.0 if r is None else r for r in results]
    all_failed = all(r is None for r in results)

    if not valid_results:
        logger.warning("No valid results. Returning zeros.")
        avg_metrics = 0.0
    else:
        avg_metrics = sum(valid_results) / len(valid_results)

    logger.info(f"Evaluation complete. Average pass@1: {avg_metrics:.3f}, All failed: {all_failed}")
    return avg_metrics, all_failed 

    
def main():
    """Main entry point to run MAS-ZERO evaluation on mbpp benchmark."""
    logger.info("Initializing models and benchmark...")

    # Optimizer LLM (for planning)
    # optimizer_config = SiliconFlowConfig(
    #     model="Pro/deepseek-ai/DeepSeek-V3.1-Terminus",
    #     siliconflow_key=SILICONFLOW_API_KEY_O,
    #     temperature=0.8,
    #     max_tokens=1000
    # )
    # optimizer_llm = SiliconFlowLLM(config=optimizer_config)

    # Executor LLM (for task execution)
    # executor_config = SiliconFlowConfig(
    #     model="Pro/deepseek-ai/DeepSeek-V3",
    #     siliconflow_key=SILICONFLOW_API_KEY_E,
    #     temperature=0.4,
    #     max_tokens=1000
    # )
    # executor_llm = SiliconFlowLLM(config=executor_config)

    optimizer_config  = OpenAILLMConfig(
        model="gpt-4o",
        openai_key= OPENAI_API_KEY_O,
        temperature=0.8,
        max_tokens=1000
        )
    optimizer_llm = OpenAILLM(config=optimizer_config)

    executor_config = OpenAILLMConfig(
        model="gpt-4o-mini",
        openai_key= OPENAI_API_KEY_E,
        temperature=0.4,
        max_tokens=1000
        )
    executor_llm = OpenAILLM(config=executor_config)

    # Initialize MAS-ZERO system
    mz = MAS_ZERO(optimizer_llm=optimizer_llm, executor_llm=executor_llm)
    benchmark = HumanEval()

    # Run async evaluation
    logger.info("Starting evaluation process...")
    avg_pass_at_1, all_failed = execute(benchmark=benchmark, mz=mz, interation_num=5)

    logger.info(f"Final Average pass@1 Score on mbpp: {avg_pass_at_1:.3f}")
    logger.info(f"All examples failed? {all_failed}")

if __name__ == "__main__":
    main()