to_sign(b::Bool) = ifelse(b, 1, -1)

function to_diff(s::T) where T
    x = 1-1/(1+s)
    x - (1 - x)
end

function valuediff(rewards::AbstractVector{Bool}, sides::AbstractVector{Bool}, γ::T) where T
    start = zero(T)
    N = length(rewards)
    @assert N == length(sides)
    vals = fill(start, N)
    for i in 1:N
        previous = i == 1 ? start : vals[i-1] 
        vals[i] = (one(T)-γ) * previous + γ * rewards[i] * to_sign(sides[i]) 
    end
    vals
end

function valuediff(rewards::AbstractVector{Bool}, sides::AbstractVector{Bool}, prob::TaskStats{T}) where T
    ratios = probabilityratio(rewards, sides, prob)
    to_diff.(ratios) .* prob.rwd
end

function probabilityratio(rewards::AbstractVector{Bool}, sides::AbstractVector{Bool}, prob::TaskStats{T}) where T
    start = one(T)
    N = length(rewards)
    vals = fill(start, N)
    reward_evidence = zero(T)
    failure_evidence = one(T) / (one(T) - prob.rwd)
    for i = 1:N
        prior = if i == 1
            start
        else
            prevlow = sides[i-1] ? one(T) / vals[i-1] : vals[i-1]
            lowprior = (prevlow + prob.dpl) / (one(T) - prob.dpl)
            (sides[i-1] == sides[i]) ? lowprior : one(T) / lowprior
        end
        evidence = rewards[i] ? reward_evidence : failure_evidence
        posterior = prior * evidence
        vals[i] = sides[i] ? one(T) / posterior : posterior
    end
    vals
end
