module ValueInferenceTools

export TaskStats, InferenceAccumulator, ValueAccumulator, Simulator
export accumulatefromdata, inferencesimulator, valuesimulator

using StatsFuns: logistic

include("probabilities.jl")
include("accumulate.jl")
include("agent.jl")

end # module
