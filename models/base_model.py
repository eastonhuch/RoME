import numpy as np
from abc import ABC, abstractmethod
from typing import Callable


class BaseTS(ABC):
    """Abstract base class for Thompson sampler"""
    featurize: Callable    
    
    @abstractmethod
    def sample_action(self, context: np.ndarray, user_idx: np.ndarray, time_idx: np.ndarray) -> np.ndarray:
        """Samples an action for the current context based on previous information

        Parameters
        ----------
        context: np.ndarray
            New context vector
        user_idx: np.ndarray
            Vector indicating the user index for each value in the context vector
        time_idx: np.ndarray
            Vector indicating the time index for each value in the context vector

        Returns
        -------
        np.ndarray
            Array of actions
        """
        pass

    @abstractmethod
    def update(self, context: np.ndarray, context_extra: np.ndarray, user_idx: np.ndarray, time_idx: np.ndarray, action: np.ndarray, reward: np.ndarray) -> 'BaseTS':
        """Updates parameter estimates based on most recent context, action, and reward

        Parameters
        ----------
        context: np.ndarray
            New context vector
        user_idx: np.ndarray
            Vector indicating the user index for each value in the context vector
        time_idx: np.ndarray
            Vector indicating the time index for each value in the context vector
        action: np.ndarray
            Vector of actions
        reward: np.ndarray
            Vector of rewards

        Returns
        -------
        np.ndarray
            Updated Thompson sampler
        """
        pass

    @abstractmethod
    def reset(self) -> 'BaseTS':
        """Resets Thompson Sampler to original state, prior to selecting actions or updating prior

        Returns
        -------
        BaseTS
            Reset Thompson sampler
        """
        pass

    def calculate_optimal_action(self, context: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Calculate the optimal action for a given context and sampled value of theta

        Parameters
        ----------
        context: np.ndarray
            2d array of contexts
        theta: np.ndarray
            2d array of user-/time-specific thetas

        Returns
        -------
        np.ndarray
            1d array of optimal actions (0s and 1s)
        """
        features = self.featurize(context)
        advantage = (features * theta).sum(axis=1)
        optimal_actions = (advantage > 0).astype(int)
        return optimal_actions
