to_sign(b::Bool) = ifelse(b, 1, -1)

function computevalue(rewards::AbstractVector{Bool}, sides::AbstractVector{Bool}, γ; start = 0.0)
    N = length(rewards)
    @assert N == length(sides)
    vals = fill(start, N)
    for i in 1:N
        previous = i == 1 ? start : vals[i-1] 
        vals[i] = (1-γ) * previous + γ * rewards[i] * to_sign(side[i]) 
    end
    vals
end
