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
    adahedge(ℓ::AbstractMatrix)

Compute the weights and losses of the AdaHedge algorithm.

This is the Julia port of the original MATLAB implementation (see [^dR14], Figure 1),

[^dR14]: S. de Rooij et al., "Follow the Leader If You Can, Hedge If You Must", Journal of Machine Learning Research, 15, 2014, 1281-1316.

# Parameters
- Loss matrix (`ℓ`): The losses incurred by the experts (columns) across rounds (rows).

# Examples
```jldoctest
julia> ℓ = [1 0; 1 0; 0 1; 1 0]
4×2 Matrix{Int64}:
 1  0
 1  0
 0  1
 1  0

julia> W, h = adahedge(ℓ);

julia> W
4×2 Matrix{Float64}:
 0.5        0.5
 0.2        0.8
 0.0848027  0.915197
 0.255116   0.744884

julia> h
4×1 Matrix{Float64}:
 0.5
 0.2
 0.9151973229650137
 0.2551156349813308
```
"""
function adahedge(ℓ::AbstractMatrix)
    T, K = size(ℓ)
    K >= 2 || throw(DomainError(K, "ℓ must have second dimension greater or equal than 2"))
    W = fill(NaN, T, K)
    h = fill(NaN, T, 1)
    L = zeros(1, K)
    Δ = 0

    for t = 1:T
        η = log(K) / Δ
        w, Mprev = mix(η, L)
        W[t, :] = w
        l = ℓ[t, :]'
        h[t] = dot(w, l)
        L += l
        _, M = mix(η, L)
        δ = max(0, h[t] - (M-Mprev))  # max clips numeric Jensen violation
        Δ += δ
    end

    return W, h
end

function mix(η, L)
    mn = minimum(L)
    if η == Inf  # limit behavior: FTL
        w = L .== mn
    else
        w = exp.(-η .* (L.-mn))
    end
    s = sum(w)
    w /= s
    M = mn - log(s/length(L))/η
    return w, M
end
