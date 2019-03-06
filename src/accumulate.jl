to_sign(b::Bool) = ifelse(b, 1, -1)

function to_diff(s::T) where T
    x = 1-1/(1+s)
    x - (1 - x)
end

updatevaluediff(rw::Bool, sd::Bool, γ::T, prev::Nothing) where {T} = updatevaluediff(rw, sd, γ)

function updatevaluediff(rw::Bool, sd::Bool, γ::T, prev::T = zero(T))::T where T
    (one(T)-γ) * prev + γ * rw * to_sign(sd)
end

function valuediff(rewards::AbstractVector{Bool}, sides::AbstractVector{Bool}, γ::T) where T
    vals = Vector{T}(undef, length(rewards))
    for (i, (rw, sd)) in enumerate(zip(rewards, sides))
        prev = get(vals, i-1, nothing)
        vals[i] = updatevaluediff(rw, sd, γ, prev)
    end
    vals
end

################################################################################

function valuediff(prob::TaskStats{T}, rewards::AbstractVector{Bool}, sides::AbstractVector{Bool}) where T
    ratios = probabilityratio(prob, rewards, sides)
    to_diff.(ratios) .* prob.rwd
end

mutable struct InferenceAccumulator{T}
    prob::TaskStats{T}
    value::T
    side::Union{Bool, Missing}
    reward_evidence::T
    failure_evidence::T
    function InferenceAccumulator(prob::TaskStats{T}, value::T = zero(T), side::Union{Bool, Missing} = missing;
                                  reward_evidence::T = zero(T),
                                  failure_evidence::T = one(T) / (one(T) - prob.rwd)) where {T}
        new{T}(prob, value, side, reward_evidence, failure_evidence)
    end
end

Base.getindex(s::InferenceAccumulator) = s.value
Base.setindex!(s::InferenceAccumulator, val) = (s.value = val; s.value)

function updateprobabilityratio!(acc::InferenceAccumulator{T}, rwd, sd)::T where T
    prob = acc.prob
    prior = if ismissing(acc.side)
        acc[]
    else
        prevlow = acc.side ? one(T) / acc[] : acc[]
        lowprior = (prevlow + prob.dpl) / (one(T) - prob.dpl)
        (acc.side == sd) ? lowprior : one(T) / lowprior
    end
    evidence = ifelse(rwd, acc.reward_evidence, acc.failure_evidence)
    posterior = prior * evidence
    acc.side = sd
    setindex!(acc, sd ? one(T) / posterior : posterior)
end

function probabilityratio(prob::TaskStats{T}, rewards::AbstractVector{Bool}, sides::AbstractVector{Bool}) where T
    acc = InferenceAccumulator(prob)
    N = length(rewards)
    vals = Vector{T}(undef, N)
    for (i, (rw, sd)) in enumerate(zip(rewards, sides))
        vals[i] = updateprobabilityratio!(acc, rw, sd)
    end
    vals
end
