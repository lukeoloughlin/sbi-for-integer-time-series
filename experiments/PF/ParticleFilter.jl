using StatsBase
using Distributions

"""
log-sum-exp trick
"""
function log_sum_exp(omega)
    os, _ = findmax(omega)
    acc = 0.0 
    for i in 1:length(omega)
        acc += exp(omega[i] - os)
    end
    return log(acc) + os
end



function inverse_cdf!(su, weights, indices)
    """inverse cdf for sorted uniform input"""
    j = 1
    s = weights[1]
    N = length(su)
    for n in 1:N
        while su[n] > s
            j += 1
            s += weights[j]
        end
        indices[n] = j
    end
end


function systematic_resample!(normalized_weights, states, indices, sorted_uniforms)
    """Descirbed in and Introduction to Sequential Monte Carlo, Chopin and Papaspiliopoulos, 2020 Page 117"""
    n = length(normalized_weights)
    sorted_uniforms .= ([i for i in 0:(n-1)] .+ rand()) ./ n
    inverse_cdf!(sorted_uniforms, normalized_weights, indices)
    states .= states[indices, :]
end