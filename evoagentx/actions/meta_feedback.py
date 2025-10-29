from pydantic import Field
from typing import List, Optional

from ..core.logging import logger
from ..models.base_model import BaseLLM
from .action import Action, ActionInput, ActionOutput
from ..prompts.meta_controller import META_FEEDBACK_ACTION   


class MetaFeedbackInput(ActionInput):
    """
    Input specification for the MetaAgent feedback action.
    """
    goal: str = Field(description="The overall goal of the workflow.")
    workflow: str = Field(description="A brief description of the entire workflow.")
    subtask_level_pairs: List[dict] = Field(description="The list of (sub-task, agent answer) pairs to evaluate.")
    agent_level_pairs: List[dict] = Field(description="The list of (agent, agent answer) pairs to evaluate.")

class MetaFeedbackOutput(ActionOutput):
    """
    Output structure for the MetaAgent feedback action.
    """
    task: str = Field(description="The name of the sub-task being evaluated.")
    agents: List[dict] = Field(description="List of agents involved in this sub-task.")
    task_feedback: str = Field(description="Feedback on whether the sub-task was solved correctly and completely.")


class MetaFeedback(Action):
    """
    Action for evaluating sub-task and agent execution results.
    The MetaAgent reviews input-output pairs, detects solvability issues, and generates
    improvement feedback for both sub-task decomposition and agent assignment.
    """

    def __init__(self, **kwargs):

        name = kwargs.pop("name") if "name" in kwargs else META_FEEDBACK_ACTION["name"]
        description = kwargs.pop("description") if "description" in kwargs else META_FEEDBACK_ACTION["description"]
        prompt = kwargs.pop("prompt") if "prompt" in kwargs else META_FEEDBACK_ACTION["prompt"]
        inputs_format = kwargs.pop("inputs_format", None) or MetaFeedbackInput
        outputs_format = kwargs.pop("outputs_format", None) or MetaFeedbackOutput
        super().__init__(name=name, description=description, prompt=prompt,
                         inputs_format=inputs_format, outputs_format=outputs_format, **kwargs)
    
    def execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, 
                sys_msg: Optional[str]=None, return_prompt: bool = False, **kwargs) -> MetaFeedbackOutput:
        """Execute the meta-feedback process.

        This method uses the provided language model to analyze (sub-task, agent answer) pairs,
        check solvability and correctness, and generate structured feedback.

        Args:
            llm: The language model to use for evaluation.
            inputs: Input data containing candidate pairs, history, and suggestions.
            sys_msg: Optional system message for the language model.
            return_prompt: Whether to return both the result and the prompt used.
            **kwargs: Additional keyword arguments.
            
        Returns:
            If return_prompt is False (default): The generated feedback.
            If return_prompt is True: A tuple of (feedback, prompt used).
            
        Raises:
            ValueError: If the inputs are None or empty.
        """
        if not inputs:
            logger.error("MetaFeedback action received invalid `inputs`: None or empty.")
            raise ValueError('The `inputs` to MetaFeedback action is None or empty.')

        inputs_format: MetaFeedbackInput = self.inputs_format
        outputs_format: MetaFeedbackOutput = self.outputs_format

        prompt_params_names = inputs_format.get_attrs()
        prompt_params_values = {param: inputs.get(param, "") for param in prompt_params_names}
        prompt = self.prompt.format(**prompt_params_values)

        message = llm.generate(
            prompt=prompt,
            system_message=sys_msg,
            parser=outputs_format,
            parse_mode="json"
        )

        if return_prompt:
            return message, prompt
        
        return message
