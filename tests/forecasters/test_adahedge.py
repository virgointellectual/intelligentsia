# Copyright 2023 VIRGO INTELLECTUAL PROPERTY LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

from intelligentsia.forecasters import adahedge


def test_wrong_number_of_experts():
    # ValueError due to wrong number of experts
    l = np.array([[1.0, 2.0]]).T
    message = 'l must have second dimension greater or equal than 2'
    with pytest.raises(ValueError, match=message):
        adahedge(l)


@pytest.mark.parametrize('K', [2, 3])
def test_equal_weights_for_identical_experts(K):
    l = np.tile(np.array([range(10)]).T, K)
    W, _ = adahedge(l)
    np.testing.assert_array_equal(W, np.full(W.shape, 1/K))
