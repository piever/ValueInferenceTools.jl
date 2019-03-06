to_sign(b::Bool) = ifelse(b, 1, -1)

function to_diff(s::T) where T
    x = 1-1/(1+s)
    x - (1 - x)
end

updatevaluediff(γ::T, rw::Bool, sd::Bool, prev::Nothing) where {T} = updatevaluediff(γ, rw, sd)

function updatevaluediff(γ::T, rw::Bool, sd::Bool, prev::T = zero(T))::T where T
    (one(T)-γ) * prev + γ * rw * to_sign(sd)
end

function valuediff(γ::T, rewards::AbstractVector{Bool}, sides::AbstractVector{Bool}) where T
    vals = Vector{T}(undef, length(rewards))
    for (i, (rw, sd)) in enumerate(zip(rewards, sides))
        prev = get(vals, i-1, nothing)
        vals[i] = updatevaluediff(γ, rw, sd, prev)
    end
    vals
end

################################################################################

function valuediff(prob::TaskStats{T}, rewards::AbstractVector{Bool}, sides::AbstractVector{Bool}) where T
    ratios = probabilityratio(prob, rewards, sides)
    to_diff.(ratios) .* prob.rwd
end

mutable struct InferenceAccumulator{T, I}
    prob::TaskStats{T}
    itr::I
    value::T
    side::Union{Bool, Nothing}
    reward_evidence::T
    failure_evidence::T
    function InferenceAccumulator(prob::TaskStats{T}, itr::I, value::T = zero(T), side::Union{Bool, Nothing} = nothing;
                                  reward_evidence::T = zero(T),
                                  failure_evidence::T = one(T) / (one(T) - prob.rwd)) where {T, I}
        new{T, I}(prob, itr, value, side, reward_evidence, failure_evidence)
    end
end

Base.getindex(s::InferenceAccumulator) = s.value
Base.setindex!(s::InferenceAccumulator, val) = (s.value = val; s.value)

function Base.iterate(acc::InferenceAccumulator, args...)
    next = iterate(acc.itr, args...)
    next === nothing && return nothing
    (val, status) = next
    rwd, sd = val
    return updateprobabilityratio!(acc, rwd, sd), status
end

Base.IteratorSize(::Type{InferenceAccumulator{T, I}}) where {T, I} = Base.IteratorSize(I)
Base.IteratorEltype(::Type{InferenceAccumulator{T, I}}) where {T, I} = Base.HasEltype()

Base.length(acc::InferenceAccumulator) = length(acc.itr)
Base.size(acc::InferenceAccumulator) = size(acc.itr)
Base.eltype(::Type{InferenceAccumulator{T, I}}) where {T, I} = T

function updateprobabilityratio!(acc::InferenceAccumulator{T}, rwd, sd)::T where T
    prob = acc.prob
    prior = if acc.side === nothing
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
    collect(InferenceAccumulator(prob, zip(rewards, sides)))
end
