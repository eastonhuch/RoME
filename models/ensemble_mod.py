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

from collections import UserList
from random import Random
from typing import Iterator

from river.base.estimator import Estimator
from river.base.wrapper import Wrapper


class Ensemble(UserList):
    """An ensemble is a model which is composed of a list of models.

    Parameters
    ----------
    models

    """

    def __init__(self, models: Iterator[Estimator]):
        super().__init__(models)

        if len(self) < self._min_number_of_models:
            raise ValueError(
                f"At least {self._min_number_of_models} models are expected, "
                + f"only {len(self)} were passed"
            )

    @property
    def _min_number_of_models(self):
        return 2

    @property
    def models(self):
        return self.data


class WrapperEnsemble(Ensemble, Wrapper):
    """A wrapper ensemble is an ensemble composed of multiple copies of the same model.

    Parameters
    ----------
    model
        The model to copy.
    n_models
        The number of copies to make.
    seed
        Random number generator seed for reproducibility.

    """

    def __init__(self, models, n_models, seed):
        super().__init__(model for model in models)
        self.model = models[0]
        self.n_models = n_models
        self.seed = seed
        self._rng = Random(seed)

    @property
    def _wrapped_model(self):
        return self.model