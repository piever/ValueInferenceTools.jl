using ValueInferenceTools
using ValueInferenceTools: to_sign
using Test
using Random

@testset "inferenceaccumulator" begin
    prob = TaskStats(0.9, 0.3)
    rews = [true, true, false, false, true]
    side = [true, true ,true, false, false]

    ratios = zeros(length(rews))
    for i in 2:length(rews)
        term = (ratios[i-1]+prob.dpl)/(1-prob.dpl)
        side[i] == side[i-1] || (term = 1/term)
        ratios[i] = rews[i] ? 0.0 : term/(1-prob.rwd)
    end
    vals = @. $(prob.rwd) * (1 - 2 / (ratios ^ (-to_sign(side)) + 1))

    @test vals ≈ accumulatefromdata(InferenceAccumulator(prob), rews, side)

    Random.seed!(4)
    rews = rand(Bool, 100)
    side = rand(Bool, 100)

    ratios = zeros(length(rews))
    for i in 2:length(rews)
        term = (ratios[i-1]+prob.dpl)/(1-prob.dpl)
        side[i] == side[i-1] || (term = 1/term)
        ratios[i] = rews[i] ? 0.0 : term/(1-prob.rwd)
    end
    vals = @. $(prob.rwd) * (1 - 2 / (ratios ^ (-to_sign(side)) + 1))
    @test vals ≈ accumulatefromdata(InferenceAccumulator(prob), rews, side)
end
