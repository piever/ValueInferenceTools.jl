# Interface

abstract type AbstractAccumulator{T}; end

Base.getindex(s::T) where {T<:AbstractAccumulator} = error("getindex not defined for $T")
Base.setindex!(s::T, val) where {T<:AbstractAccumulator} = error("setindex! not defined for $T")

function update!(acc::T, rw, sd) where T<:AbstractAccumulator
    error("update! not defined for $T")
end

################################################################################

to_sign(b::Bool) = ifelse(b, 1, -1)

mutable struct ValueAccumulator{T} <: AbstractAccumulator{T}
    γ::T
    value::T
end

ValueAccumulator(γ::T) where {T} = ValueAccumulator(γ, zero(T))

Base.getindex(s::ValueAccumulator) = s.value
Base.setindex!(s::ValueAccumulator, val) = (s.value = val; s.value)

function update!(acc::ValueAccumulator{T}, rw, sd)::T where T
    setindex!(acc, (one(T)-acc.γ) * acc[] + acc.γ * rw * to_sign(sd))
end

################################################################################

function to_diff(s::T)::T where T
    x = 1-1/(1+s)
    x - (1 - x)
end

function to_ratio(s::T)::T where T
    x = (s+1)/2
    1 / (1-x) - 1
end

mutable struct InferenceAccumulator{T} <: AbstractAccumulator{T}
    prob::TaskStats{T}
    value::T
    side::Union{Bool, Nothing}
    reward_evidence::T
    failure_evidence::T
    ratio::T
    function InferenceAccumulator(prob::TaskStats{T}, value::T = zero(T), side::Union{Bool, Nothing} = nothing;
                                  reward_evidence::T = zero(T),
                                  failure_evidence::T = one(T) / (one(T) - prob.rwd)) where {T}
        new{T}(prob, value, side, reward_evidence, failure_evidence, to_ratio(value/prob.rwd))
    end
end

Base.getindex(s::InferenceAccumulator) = s.value
Base.setindex!(s::InferenceAccumulator, val) = (s.value = val; s.value)

function update!(acc::InferenceAccumulator{T}, rwd, sd)::T where T
    prob = acc.prob
    ratio = acc.ratio
    prior = if acc.side === nothing
        ratio
    else
        prevlow = acc.side ? one(T) / ratio : ratio
        lowprior = (prevlow + prob.dpl) / (one(T) - prob.dpl)
        (acc.side == sd) ? lowprior : one(T) / lowprior
    end
    evidence = ifelse(rwd, acc.reward_evidence, acc.failure_evidence)
    posterior = prior * evidence
    acc.side = sd
    acc.ratio = sd ? one(T) / posterior : posterior
    setindex!(acc, to_diff(acc.ratio)*prob.rwd)
end

###############################################################################

struct AccumulateFromData{T, I, W<:AbstractAccumulator{T}}
    acc::W
    itr::I
    AccumulateFromData(acc::AbstractAccumulator{T}, itr::I) where {T, I} =
        new{T, I, typeof(acc)}(acc, itr)
end

getiterator(s::AccumulateFromData) = s.itr
getaccumulator(s::AccumulateFromData) = s.acc

function Base.iterate(acc::AccumulateFromData, args...)
    next = iterate(getiterator(acc), args...)
    next === nothing && return nothing
    (val, status) = next
    rwd, sd = val
    return update!(getaccumulator(acc), rwd, sd), status
end

Base.IteratorSize(::Type{AccumulateFromData{T, I}}) where {T, I} = Base.IteratorSize(I)
Base.IteratorEltype(::Type{AccumulateFromData{T, I}}) where {T, I} = Base.HasEltype()

Base.length(acc::AccumulateFromData) = length(getiterator(acc))
Base.size(acc::AccumulateFromData) = size(getiterator(acc))
Base.eltype(::Type{<:AccumulateFromData{T, I}}) where {T, I} = T

Base.getindex(s::AccumulateFromData) = getindex(getaccumulator(s))
Base.setindex!(s::AccumulateFromData, val) = setindex!(getaccumulator(s), val)

accumulatefromdata(acc::AbstractAccumulator, rewards, sides) =
    collect(AccumulateFromData(acc, zip(rewards, sides)))

function valuediff(γ::T, rewards::AbstractVector{Bool}, sides::AbstractVector{Bool}) where T
    accumulatefromdata(ValueAccumulator(γ), rewards, sides)
end

function valuediff(prob::TaskStats{T}, rewards::AbstractVector{Bool}, sides::AbstractVector{Bool}) where T
    ratios = probabilityratio(prob, rewards, sides)
    to_diff.(ratios) .* prob.rwd
end

function probabilityratio(prob::TaskStats{T}, rewards::AbstractVector{Bool}, sides::AbstractVector{Bool}) where T
    accumulatefromdata(InferenceAccumulator(prob), rewards, sides)
end
