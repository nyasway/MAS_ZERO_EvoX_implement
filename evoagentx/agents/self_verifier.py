from .agent import Agent
from ..actions.self_verification import SelfVerification
from ..prompts.self_verifier import SELF_VERIFIER


class SelfVerifier(Agent):
    """An agent responsible for verifying and selecting the most appropriate final answer 
    from a list of candidate answers across iterations.
    
    The SelfVerifier agent applies three rules:
        1. Frequency Priority – Prefer answers that appear most frequently.
        2. Invalid Answer Filtering – Discard nonsensical or out-of-scope answers.
        3. Final Judgment – As a verifier, select the best single valid answer.
    
    Attributes:
        name (str): Name of the self verifier agent, defaults to the value in SELF_VERIFIER
        description (str): Description of the agent's purpose and capabilities, defaults to SELF_VERIFIER
        system_prompt (str): System prompt guiding the agent's behavior, defaults to SELF_VERIFIER
        actions (List[Action]): List of actions the agent can perform, defaults to [SelfVerifier()]
    """
    def __init__(self, **kwargs):

        name = kwargs.pop("name") if "name" in kwargs else SELF_VERIFIER["name"]
        description = kwargs.pop("description") if "description" in kwargs else SELF_VERIFIER["description"]
        system_prompt = kwargs.pop("system_prompt") if "system_prompt" in kwargs else SELF_VERIFIER["system_prompt"]
        actions = kwargs.pop("actions") if "actions" in kwargs else [SelfVerification()]
        super().__init__(name=name, description=description, system_prompt=system_prompt, actions=actions, **kwargs)

    @property
    def self_verification_action_name(self):
        """Get the name of the SelfVerification action associated with this agent.
        
        Returns:
            The name of the SelfVerification action in this agent's action registry
        """
        return self.get_action_name(action_cls=SelfVerification)
