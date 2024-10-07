# This section of the code is modified from the River ML library: https://riverml.xyz/
# As such, this portion of the code is licensed under the BSD 3-Clause License
# BSD 3-Clause License

# Copyright (c) 2020, the river developers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from river import base, linear_model, utils
import numpy as np
from models.ensemble_mod import WrapperEnsemble

class BaseBagging(WrapperEnsemble):
    def learn_one(self, x, y, subsample: float = 0.5, **kwargs):
        # subsample <= 0. performs the bootstrap
        # Otherwise, observations are included in a tree with probability = subsample

        for model in self:
            if subsample <= 0:
                poisson_val = utils.random.poisson(1, self._rng)
            else:
                poisson_val = int(self._rng.random() < subsample)
            if not hasattr(model, "poisson_vals"):
                model.poisson_vals = []
            model.poisson_vals.append(poisson_val)
            for _ in range(poisson_val):
                model.learn_one(x, y, **kwargs)

        return self


class BaggingRegressor(BaseBagging, base.Regressor):
    """Online bootstrap aggregation for regression.

    For each incoming observation, each model's `learn_one` method is called `k` times where
    `k` is sampled from a Poisson distribution of parameter 1. `k` thus has a 36% chance of
    being equal to 0, a 36% chance of being equal to 1, an 18% chance of being equal to 2, a 6%
    chance of being equal to 3, a 1% chance of being equal to 4, etc. You can do
    `scipy.stats.utils.random.poisson(1).pmf(k)` for more detailed values.

    Parameters
    ----------
    model
        The regressor to bag.
    n_models
        The number of models in the ensemble.
    seed
        Random number generator seed for reproducibility.

    Examples
    --------

    In the following example three logistic regressions are bagged together. The performance is
    slightly better than when using a single logistic regression.

    >>> from river import datasets
    >>> from river import ensemble
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import optim
    >>> from river import preprocessing

    >>> dataset = datasets.TrumpApproval()

    >>> model = preprocessing.StandardScaler()
    >>> model |= ensemble.BaggingRegressor(
    ...     model=linear_model.LinearRegression(intercept_lr=0.1),
    ...     n_models=3,
    ...     seed=42
    ... )

    >>> metric = metrics.MAE()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 0.68886

    References
    ----------
    [^1]: [Oza, N.C., 2005, October. Online bagging and boosting. In 2005 IEEE international conference on systems, man and cybernetics (Vol. 3, pp. 2340-2345). Ieee.](https://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf)

    """

    def __init__(self, model: base.Regressor, n_models=10, seed: int = None, subsample: float = 0.5):
        self.subsample = subsample

        # Randomly change radius
        models = [model.clone() for _ in range(n_models)]
        if hasattr(model, "feature_quantizer"):
            for model_i in models:
                model_i.feature_quantizer.radius = 0.1 + np.random.exponential(0.1)
                model_i.feature_quantizer.std_prop = 0.1 + np.random.exponential(0.1)

        super().__init__(models, n_models, seed)

    @classmethod
    def _unit_test_params(cls):
        yield {"model": linear_model.LinearRegression()}

    def learn_one(self, x, y, subsample: float = 0.5, **kwargs):
        subsample = self.subsample if subsample is None else subsample
        return super().learn_one(x, y, subsample, **kwargs)

    def predict_one(self, x, i=None, **kwargs):
        """Averages the predictions of each regressor."""
        prediction_sum = 0.
        count = 0
        for regressor in self:
            if (i is None) or (i < 0) or (i >= len(regressor.poisson_vals)) or (regressor.poisson_vals[i] == 0):
                count += 1
                prediction_sum += regressor.predict_one(x, **kwargs)
        prediction = prediction_sum / count if count else 0.
        return prediction