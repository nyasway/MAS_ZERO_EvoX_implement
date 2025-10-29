from .agent import Agent
from ..actions.meta_feedback import MetaFeedback
from ..prompts.meta_controller import META_CONTROLLER 


class MetaController(Agent):
    """An agent responsible for evaluating sub-task and agent execution results 
    and providing structured feedback to improve task decomposition and agent assignment.

    The MetaController agent applies three key evaluation mechanisms:
        1. Explicit Refusal Signal – Detect `[TOO HARD]` tags and mark sub-tasks as unsolvable in current form.
        2. Input-Output Coherence – Identify empty, illogical, or invalid outputs.
        3. Independent Completeness – Ensure each sub-task provides a complete and standalone answer.

    Based on these evaluations, the agent produces two types of improvement feedback:
        - Subtask Revision Feedback
        - Agent Assignment Feedback

    Attributes:
        name (str): Name of the meta controller agent, defaults to the value in META_AGENT
        description (str): Description of the agent's purpose and capabilities, defaults to META_AGENT
        system_prompt (str): System prompt guiding the agent's behavior, defaults to META_AGENT
        actions (List[Action]): List of actions the agent can perform, defaults to [MetaFeedback()]
    """
    def __init__(self, **kwargs):

        name = kwargs.pop("name") if "name" in kwargs else META_CONTROLLER["name"]
        description = kwargs.pop("description") if "description" in kwargs else META_CONTROLLER["description"]
        system_prompt = kwargs.pop("system_prompt") if "system_prompt" in kwargs else META_CONTROLLER["system_prompt"]
        actions = kwargs.pop("actions") if "actions" in kwargs else [MetaFeedback()]
        super().__init__(name=name, description=description, system_prompt=system_prompt, actions=actions, **kwargs)

    @property
    def meta_feedback_action_name(self):
        """Get the name of the MetaFeedback action associated with this agent.
        
        Returns:
            The name of the MetaFeedback action in this agent's action registry
        """
        return self.get_action_name(action_cls=MetaFeedback)
