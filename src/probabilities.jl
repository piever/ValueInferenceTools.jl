struct TaskStats{T}
    rwd::T
    dpl::T
end

function no_reward(prob::TaskStats)
    p, g = prob.rwd, prob.dpl
    return (1-p)*g / (1-(1-p)*(1-g))
end

function no_reward(prob::TaskStats, n)
    @assert n >= 0
    p, g = prob.rwd, prob.dpl
    return n == 0 ? 1.0 : (1-p)*(g+(1-g)*no_reward(prob, n-1))
end

function no_more_reward(prob::TaskStats, args...)
    g = prob.dpl
    g+(1-g)*no_reward(prob, args...)
end

function high_sequence_probability(prob::TaskStats, res::AbstractArray{Bool})
    p, g = prob.rwd, prob.dpl
    n = length(res)
    r = count(res)
    o = n-r
    return p^r*(1-p)^o*(1-g)^(n-1)
end

function sequence_probability(prob::TaskStats, res::AbstractArray{Bool})
    p, g = prob.rwd, prob.dpl
    ls = findlast(res)
    ls === nothing && return no_reward(prob)
    ph = high_sequence_probability(prob, res[1:ls])
    return ph*no_more_reward(prob, length(res)-ls)
end

one_reward(prob::TaskStats) = prob.rwd*no_more_reward(prob)


