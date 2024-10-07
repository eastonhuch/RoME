import numpy as np


class DataGenerator:
    """Class for generating data along diagonals of rectangular n by t array"""
    finished: bool = False

    def __init__(
            self, n_max: int = None, t_max: int = None, stage_max: int = None,
            assumptions: str = "nonlinear", random_seed: int = 1, nonlinear_scale: float = 1.,
            linear_baseline_coef = None
            ) -> 'DataGenerator':
        """Initializes class of given size with baseline causal effects
        
        Parameters
        ----------
        n_max: int, default=None
            The number of users. If None, then stage_max must be specified.
        t_max: int, default=None
            The number of time points for each user.  If None, then stage_max must be specified.
        stage_max: int, default=None
            The total number of "stages"; i.e., time points where an additional participant is added.
            Specifying this parameter results in a triangular data array, as opposed to a rectangular data array.
        assumptions: str, default="nonlinear"
            What assumptions to make about the data-generating process; options are
                - nonlinear: Assumptions in paper
                - homogeneous: identical users, linear baseline, no time effects
                - hetergeneous: different users, linear baseline, no time effects
        random_seed: int, default=1
            Random seed for setting up baseline and parameters
        nonlinear_scale: float, default=1.
            A scaling factor for the nonlinear baseline
        linear_baseline_coef: default=None
            The coefficients for the linear baseline; must be specified when assumptions != "nonlinear"
        """
        if n_max is not None:
            self.n_max = n_max
            if t_max is None:
                raise ValueError("If n_max is specified, t_max must also specified")
            self.t_max = t_max
            self.stage_max = t_max + (n_max-1)
            self.array_type = "rectangular"
        elif stage_max is None:
            raise ValueError("If n_max is not specified, stage_max must be specified")
        else:
            self.stage_max = stage_max
            self.t_max = stage_max
            self.n_max = stage_max
            self.array_type = "triangular"
    
        self.stage = 0
        self.all_users_idx = np.array(range(self.n_max))
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        if assumptions not in ("nonlinear", "homogeneous", "heterogeneous"):
            raise ValueError("assumptions must be 'nonlinear', 'standard', or 'heterogeneous'")
        self.assumptions = assumptions
        if self.assumptions == "nonlinear":
            self.normal_vars = np.random.normal(size=(20, 100))
            self.uniform_vars = np.random.uniform(size=(1000,))
        else:
            if linear_baseline_coef is None:
                raise ValueError("linear_baseline_coef must be specified when assumptions != 'nonlinear'")
        self.linear_baseline_coef = linear_baseline_coef
        self.nonlinear_scale = nonlinear_scale


    def sample_theta(self, theta_base: np.ndarray, theta_time_init: np.ndarray = None) -> 'DataGenerator':
        """Randomly samples values for theta_user and theta_time

        Parameters
        ----------
        theta_base: np.ndarray
            1d array of baseline coefficients for advantage function
        theta_time_init: np.ndarray, default=None
            Initial values for the theta_time parameters; only relevent for assumptions == "nonlinear"

        Returns
        -------
        DataGenerator
            Updated DataGenerator object with the following values set:
            theta_base, p, theta_user, theta_time, theta
        """
        # theta_base
        self.theta_base = theta_base
        self.p = theta_base.size

        # theta_user
        if self.assumptions != "homogeneous":
            np.random.seed(self.random_seed)
            self.theta_user = np.random.normal(scale=1., size=(self.n_max, self.p))
        else:
            self.theta_user = np.zeros((self.n_max, self.p))


        # theta_base_plus_user
        self.theta_base_plus_user = self.theta_user + self.theta_base

        # theta_time
        if self.assumptions == "nonlinear":
            if theta_time_init is None:
                raise ValueError("theta_time_init must be specified for nonlinear setting")
            theta_time = []
            np.random.seed(self.random_seed)
            for t in range(self.t_max):
                theta_time_t = theta_time_init / (1 + 6*t/self.t_max) + np.random.normal(scale=0.2, size=self.p)
                theta_time.append(theta_time_t)
            self.theta_time = np.asarray(theta_time)
        else:
            self.theta_time = np.zeros((self.t_max, self.p))

        # theta
        self.theta = np.concatenate([
            self.theta_base,
            self.theta_user.flatten(),
            self.theta_time.flatten()])

        return self

    def generate_all_contexts(self, context_dim=2, extra_context_dim=2) -> 'DataGenerator':
        """Generates the context variables for all users and time points and then resets indices with the reset method"""
        self.context = np.random.uniform(low=-1., high=1., size=(self.n_max, self.t_max, context_dim))
        self.context_extra = np.random.uniform(low=-1., high=1., size=(self.n_max, self.t_max, extra_context_dim))  # Used for ML model training
        self.context_dim = context_dim
        self.extra_context_dim = extra_context_dim
        self.reset()
        return self
  
    def generate_baselines_for_nn(self) -> tuple[np.ndarray, np.ndarray]:
        """Generates baseline outcomes for training a neural network"""
        context_2d = self.context.copy().reshape((-1, self.context.shape[2]))
        if self.assumptions == "nonlinear":
            baseline = self._nonlinear_baseline_function(context_2d, scale=self.nonlinear_scale)
        else:
            baseline = self._standard_baseline_function(context_2d)
        context_extra_2d = self.context_extra.copy().reshape((-1, self.context_extra.shape[2]))
        context_full_2d = np.hstack([context_2d, context_extra_2d])
        random_error = np.random.normal(size=baseline.size)
        reward = baseline + random_error
        return context_full_2d, reward

    def update_current_context(self) -> 'DataGenerator':
        """Updates the context for the given stage

        Returns
        -------
        DataGenerator
            Updated DataGenerator object with the context values set
        """
        raw_time_idx = self.stage - self.all_users_idx
        include = (0 <= raw_time_idx) & (raw_time_idx < self.t_max)
        self.stage_time_idx = raw_time_idx[include]
        self.stage_user_idx = self.all_users_idx[include]
        self.stage_context = self.context[self.stage_user_idx, self.stage_time_idx]
        self.stage_context_extra = self.context_extra[self.stage_user_idx, self.stage_time_idx]
        return self

    def get_current_context(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gets context for current diagonal

        Returns
        -------
        np.ndarray
            New context vector
        np.ndarray
            New context_extra vector (for ML model)
        np.ndarray
            Vector indicating the user index for each value in the context vector
        np.ndarray
            Vector indicating the time index for each value in the context vector
        """
        context_tuple = (
            self.stage_context.copy(),
            self.stage_context_extra.copy(),
            self.stage_user_idx.copy(),
            self.stage_time_idx.copy())
        return context_tuple

    def play_action(self, action: np.ndarray, update_stage: bool = True) -> np.ndarray:
        """Plays an action and returns the resulting reward

        Parameters
        ----------
        action: np.ndarray
            Vector of actions
        update_stage: bool, default=True
            Whether the stage should be updated

        Returns
        -------
        np.ndarray
            Vector of rewards
        """
        # Baseline
        if self.assumptions == "nonlinear":
            baseline = self._nonlinear_baseline_function(self.stage_context, scale=self.nonlinear_scale)
        else:
            baseline = self._standard_baseline_function(self.stage_context)

        # Advantage function (causal effects)
        features = self._featurize(self.stage_context)
        stage_theta_base_plus_user = self.theta_base_plus_user[self.stage_user_idx]
        stage_theta_time = self.theta_time[self.stage_time_idx]
        stage_theta = stage_theta_base_plus_user + stage_theta_time
        potential_advantage = (features * stage_theta).sum(axis=1)
        advantage = action * potential_advantage

        # Standard normal error
        random_error = np.random.normal(size=baseline.size)

        # Compute full reward
        reward = baseline + advantage + random_error

        # Increment stage and update context
        if update_stage:
            self.increment_stage()

        return reward

    def increment_stage(self) -> 'DataGenerator':
        """Increments the stage and updates context

        Returns
        -------
        DataGenerator
            Updated DataGenerator object with incremented stage and updated context
        """
        self.stage += 1
        self.finished = self.stage >= self.stage_max
        if not self.finished:
            self.update_current_context()
        return self

    def reset(self) -> 'DataGenerator':
        """Resets the DataGenerator object to its state before returning any values in generate_all_contexts() or play_action()

        Returns
        -------
        DataGenerator
            The rest DataGenerator object
        """
        self.stage = 0
        self.finished = False
        self.update_current_context()
        return self

    @staticmethod
    def _featurize(context: np.ndarray) -> np.ndarray:
        """Static method that creates feature vectors from the given context
        
        Parameters
        ----------
        context: np.ndarray
            1d or 2d array of context vector(s)

        Returns
        -------
        np.ndarray
            2d array of feature vectors of shape (n_obs, p)
        """
        if len(context.shape) == 1:
            context_dim = context.shape[0]
        elif len(context.shape) == 2:
            context_dim = context.shape[1]
        else:
            raise ValueError("context must be 1d or 2d")
        context = context.reshape((-1, context_dim))
        n_obs = context.shape[0]
        features = np.ones((n_obs, 1 + context_dim))
        features[:, 1:] = context
        return features


    def _nonlinear_baseline_function(self, context: np.ndarray, scale: float = 1.) -> np.ndarray:
        """Method that return the nonlinear baseline reward.
        The baseline reward is the sum of randomly scaled PDFs for rotated and shifted normal random variables.
        
        Returns
        -------
        np.ndarray
            Nonlinear baseline reward
        """
        n_obs = context.shape[0]
        context1 = context[:, 0].copy()
        context2 = context[:, 1].copy()

        # Create some hard change points via recursive partitioning
        baseline = self.__recursive_split(
            np.array([True]*n_obs), context1, context2, -1., 1.,
            -1., 1., self.uniform_vars, 0, min_size=0.4, scale=scale)

        # Add smooth component on top
        for i in range(self.normal_vars.shape[0]):
            normal_iter = iter(self.normal_vars[i])
            x1 = next(normal_iter) * context1 + next(normal_iter) * context2
            x2 = next(normal_iter) * context1 + next(normal_iter) * context2
            baseline += scale * 2. * next(normal_iter) * np.exp(
                -(x1 - next(normal_iter))**2
                -(x2 - next(normal_iter))**2
            )
        return baseline

    def _standard_baseline_function(self, context: np.ndarray) -> np.ndarray:
        """Returns linear baseline reward
        
        Static method that returns a function for calculating the standard baseline reward.
        The baseline reward is a linear function of the context.
        
        Returns
        -------
        np.ndarray
            Linear reward
        """
        features = self._featurize(context)
        baseline = features @ self.linear_baseline_coef
        return baseline

    def __recursive_split(
        self, in_region: np.ndarray, x: np.ndarray, y: np.ndarray, xl: float, xr: float,
        yl: float, yr: float, u: np.ndarray, i: int, min_size: float = 0.1, scale: float = 1.):

        if min(abs(xl-xr), abs(yl-yr)) < min_size:
            return np.zeros_like(in_region)
        
        u1, u2, u3 = u[i], u[i+1], u[i+2]

        if u1 < 0.5:
            split = u2 * xl + (1.-u2) * xr
            x_below_split = x < split
            in_region_l = in_region & x_below_split
            in_region_r = in_region & (~x_below_split)
            values_increment_l = self.__recursive_split(in_region_l, x, y, xl, split, yl, yr, u, i+3, min_size=min_size, scale=scale)
            values_increment_r = self.__recursive_split(in_region_l, x, y, split, xr, yl, yr, u, i+4, min_size=min_size, scale=scale)
        else:
            split = u2 * yl + (1.-u2) * yr
            y_below_split = y < split
            in_region_l = in_region & y_below_split
            in_region_r = in_region & (~y_below_split)
            values_increment_l = self.__recursive_split(in_region_l * y_below_split, x, y, xl, xr, yl, split, u, i+3, min_size=min_size, scale=scale)
            values_increment_r = self.__recursive_split(in_region_r * (~y_below_split), x, y, xl, xr, split, yr, u, i+4, min_size=min_size, scale=scale)

        values = scale * 12. * (u3 - 0.5) * in_region.astype(float)
        values += (values_increment_l.astype(float) + values_increment_r.astype(float))
        return values