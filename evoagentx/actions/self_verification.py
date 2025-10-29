from pydantic import Field
from typing import Optional, List

from ..core.logging import logger
from ..models.base_model import BaseLLM
from .action import Action, ActionInput, ActionOutput
from ..prompts.self_verifier import SELF_VERIFICATION_ACTION


class SelfVerificationInput(ActionInput):
    """
    Input specification for the SelfVerification action.
    """
    goal: str = Field(description="The user's goal that the final answer should align with.")
    candidate_answers: List[str] = Field(description="A list of candidate answers collected from multiple iterations.")
   
class SelfVerificationOutput(ActionOutput):
    """
    Output structure for the SelfVerification action.
    """
    thoughts: Optional[str] = Field(default=None, description="The reasoning process behind selecting the final answer.")
    final_answer: str = Field(description="The single most appropriate final answer selected from the candidate list.")

class SelfVerification(Action):
    """
    Action for verifying candidate answers and selecting the best final answer.
    This class applies frequency analysis, invalid answer filtering, and final judgment rules.
    """

    def __init__(self, **kwargs):

        name = kwargs.pop("name", SELF_VERIFICATION_ACTION["name"])
        description = kwargs.pop("description", SELF_VERIFICATION_ACTION["description"])
        prompt = kwargs.pop("prompt", SELF_VERIFICATION_ACTION["prompt"])
        inputs_format = kwargs.pop("inputs_format", None) or SelfVerificationInput
        outputs_format = kwargs.pop("outputs_format", None) or SelfVerificationOutput

        super().__init__(
            name=name,
            description=description,
            prompt=prompt,
            inputs_format=inputs_format,
            outputs_format=outputs_format,
            **kwargs
        )

    def execute(
        self,
        llm: Optional[BaseLLM] = None,
        inputs: Optional[dict] = None,
        sys_msg: Optional[str] = None,
        return_prompt: bool = False,
        **kwargs
    ) -> SelfVerificationOutput:
        """
        Execute the verification process.

        This method uses the provided language model to:
        1. Apply frequency-based selection among candidate answers.
        2. Filter out invalid or nonsensical answers.
        3. Make a final judgment to select the best single answer.

        Args:
            llm: The language model to use for verification.
            inputs: Input data containing the candidate list and optional context.
            sys_msg: Optional system message for the language model.
            return_prompt: Whether to return both the result and the prompt used.
            **kwargs: Additional keyword arguments.

        Returns:
            If return_prompt is False (default): The chosen final answer.
            If return_prompt is True: A tuple of (final answer, prompt used).

        Raises:
            ValueError: If the inputs are None or empty.
        """
        if not inputs:
            logger.error("SelfVerifier action received invalid `inputs`: None or empty.")
            raise ValueError("The `inputs` to SelfVerifier action is None or empty.")

        inputs_format: SelfVerificationInput = self.inputs_format
        outputs_format: SelfVerificationOutput = self.outputs_format

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
