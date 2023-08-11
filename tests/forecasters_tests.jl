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

Random.seed!(1234)

@testset "AdaHedge" begin
    @testset "Less than two experts" begin
        ℓ = rand(10, 1)
        @test_throws DomainError adahedge(ℓ)
    end
    @testset "Equal weights for identical experts" begin
        x = rand(10, 1)
        ℓ = [x x]
        W, _ = adahedge(ℓ)
        @test W ≈ repeat([0.5], 10, 2)
    end
    @testset "Weights concentrate on clear best expert" begin
        ℓ = [zeros(100, 1) ones(100, 1)]
        W, _ = adahedge(ℓ)
        @test W[end, 1] ≈ 1
    end
end
