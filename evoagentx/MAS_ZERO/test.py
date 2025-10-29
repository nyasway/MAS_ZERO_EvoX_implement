import os 
from dotenv import load_dotenv

from evoagentx.agents import AgentManager
from evoagentx.models import SiliconFlowConfig, SiliconFlowLLM, OpenAILLMConfig, OpenAILLM
from evoagentx.MAS_ZERO.MAS_ZERO import MAS_ZERO

# Configure the model
load_dotenv()
OPENAI_API_KEY_O = os.getenv("OPENAI_API_KEY_O")
OPENAI_API_KEY_E = os.getenv("OPENAI_API_KEY_E")
SILICONFLOW_API_KEY_O = os.getenv("SILICONFLOW_API_KEY_O")
SILICONFLOW_API_KEY_E = os.getenv("SILICONFLOW_API_KEY_E")

# optimizer_config  = OpenAILLMConfig(
#     model="gpt-4o",
#     openai_key= OPENAI_API_KEY_O,
#     temperature=0.8,
#     max_tokens=1000)
# optimizer_llm = OpenAILLM(config=optimizer_config)

# executor_config = OpenAILLMConfig(
#     model="gpt-4o-mini",
#     openai_key= OPENAI_API_KEY_E,
#     temperature=0.4,
#     max_tokens=1000
#     )
# executor_llm = OpenAILLM(config=executor_config)

optimizer_config = SiliconFlowConfig(
        model="Pro/deepseek-ai/DeepSeek-V3.1-Terminus",
        siliconflow_key=SILICONFLOW_API_KEY_O,
        temperature=0.8,
        max_tokens=1000
    )
optimizer_llm = SiliconFlowLLM(config=optimizer_config)

# Executor LLM (for task execution)
executor_config = SiliconFlowConfig(
        model="Pro/deepseek-ai/DeepSeek-V3",
        siliconflow_key=SILICONFLOW_API_KEY_E,
        temperature=0.4,
        max_tokens=1000
    )
executor_llm = SiliconFlowLLM(config=executor_config)

# Initialize the model
mz = MAS_ZERO(optimizer_llm=optimizer_llm, executor_llm=executor_llm)
goal = "Let ABCDEF be a convex equilateral hexagon in which all pairs of opposite sides are parallel. The triangle whose sides are extensions of segments AB, CD, and EF has side lengths 200, 240, and 300. Find the side length of the hexagon.."
# goal = "What is the capital of France?"

output = mz.execute(goal=goal, iteration_num=5)

print(output)