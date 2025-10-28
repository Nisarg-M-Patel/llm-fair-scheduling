import numpy as np

from vidur.request_generator.base_client_assignment_generator import BaseClientAssignmentGenerator
from vidur.config import BinomialClientAssignmentGeneratorConfig


class BinomialClientAssignmentGenerator(BaseClientAssignmentGenerator):
    """
    A client assignment generator that randomly assigns between two clients (0 and 1)
    with a controllable activation rate.
    
    Parameters:
    -----------
    activation_rate : float
        The probability of selecting client 1 (0 <= activation_rate <= 1)
        activation_rate = 0.0 means always select client 0
        activation_rate = 1.0 means always select client 1
    seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, config: BinomialClientAssignmentGeneratorConfig):
        super().__init__(config)

        if not 0 <= config.activation_rate <= 1:
            raise ValueError("Activation rate must be between 0 and 1")
            
        self._activation_rate = config.activation_rate
        self._rng = np.random.RandomState(config.seed)
    
    def get_next_client_id(self):
        """
        Generate the next client ID (0 or 1) based on the activation rate.
        
        Returns:
        --------
        int
            0 with probability (1 - activation_rate)
            1 with probability activation_rate
        """
        return int(self._rng.random() < self._activation_rate)
