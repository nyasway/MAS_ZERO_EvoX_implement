import os 
from dotenv import load_dotenv
from tqdm import tqdm

from evoagentx.frameworks.multi_agent_debate.debate import MultiAgentDebateActionGraph
from evoagentx.core.logging import logger
from evoagentx.models import OpenAILLMConfig, OpenAILLM, SiliconFlowConfig, SiliconFlowLLM
from evoagentx.benchmark import benchmark, MATH

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
    permutation = np.random.permutation(len(benchmark._test_data))
    full_test_data = benchmark._test_data
    data = [full_test_data[idx] for idx in permutation[50:150]]

    if not data:
        logger.warning("No data to evaluate. Returning zeros.")
        return 0.0, True
    
    results = []
    iterator = tqdm(data, desc=f"Evaluating {benchmark.name} problems", ncols=100) if verbose else data

    for example in iterator:
        goal = example["problem"]
        instr = "Your final answer must be formatted exactly as in the simplified numeric or symbolic expression, without spaces. Do not include words, sentences, or LaTeX formatting. For example, output 'y=x+3' instead of 'y = x + 3'."
        goal = f"{goal}\n\n{instr}\n\nAnswer:"
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
            solve_rate = metrics.get("solve_rate", 0.0)
            logger.info(f"Evaluated question: {goal[:50]}... | Prediction: {prediction} | Label: {label} | Solve Rate: {solve_rate:.3f}")
            results.append(solve_rate)

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

    logger.info(f"Evaluation complete. Average Solve Rate: {avg_metrics:.3f}, All failed: {all_failed}")
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

    benchmark = MATH()

    # Run async evaluation
    logger.info("Starting evaluation process...")
    avg_solve_rate, all_failed = execute(benchmark=benchmark, debate=debate)

    logger.info(f"Final Average Solve Rate on MATH: {avg_solve_rate:.3f}")
    logger.info(f"All examples failed? {all_failed}")

if __name__ == "__main__":
    main()