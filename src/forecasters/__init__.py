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

"""
``intelligentsia.forecasters``
==============================

This module implements online learning algorithms to be used
specifically in the context of prediction with expert advice.

Algorithms present in intelligentsia.forecasters are listed below.

Parameter-free algorithms
-------------------------

    adahedge

"""

from intelligentsia.forecasters import adahedge

__all__ = adahedge.__all__.copy()

from intelligentsia.forecasters.adahedge import *
