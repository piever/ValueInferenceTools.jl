struct Simulator{T, W<:AbstractAccumulator{T}, F}
    prob::TaskStats{T}
    acc::W
    decision::F
    Simulator(p::TaskStats{T}, a::AbstractAccumulator{T}, d::F) where {T, F} = new{T, typeof(a), F}(p, a, d)
end

getaccumulator(s::Simulator) = s.acc
TaskStats(s::Simulator) = s.prob

Base.IteratorSize(::Type{<:Simulator}) = Base.IsInfinite()
Base.IteratorEltype(::Type{<:Simulator}) = Base.HasEltype()

Base.eltype(::Type{<:Simulator{T}}) where {T} =
    NamedTuple{(:value, :reward, :side, :sidehigh), Tuple{T, Bool, Bool, Bool}}

function Base.iterate(s::Simulator, (side, sidehigh) = (rand(Bool), rand(Bool)))
    prob = TaskStats(s)
    rwd = (sidehigh == side) ? (rand() < prob.rwd) : false
    val = update!(getaccumulator(s), rwd, side)
    outcome = (value = val, reward = rwd, side = side, sidehigh = sidehigh)
    (side == sidehigh) && (rand() < prob.dpl) && (sidehigh = !sidehigh)
    newside = s.decision(val, side)
    return outcome, (newside, sidehigh)
end

function inferencesimulator(prob::TaskStats, ind, β)
    acc = InferenceAccumulator(prob)
    decision = DiffRule(ind, β)
    Simulator(prob, acc, decision)
end

function valuesimulator(prob::TaskStats, γ, ind, β)
    acc = ValueAccumulator(γ)
    decision = DiffRule(ind, β)
    Simulator(prob, acc, decision)
end

struct DiffRule{T}
    ind::T
    β::T
end

DiffRule(a, b) = DiffRule(promote(a, b)...)

function (dr::DiffRule)(diff, side)
    Δind = diff + dr.ind*to_sign(side)
    return rand() < logistic(dr.β*Δind)
end
