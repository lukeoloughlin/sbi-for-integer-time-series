using DelimitedFiles
using LinearAlgebra
using Random, Distributions, AdvancedMH, MCMCChains
using Plots, StatsBase, StatsPlots
include("../ParticleFilter.jl")


function PP_forward!(b, d1, d2, p1, p2, state, log_weights, obs, q)

    num_particles = size(state, 1)

    Threads.@threads for i in 1:num_particles

        @views Y = state[i, :]
        t = 0
        rates = zeros(4)

        while true
            pred = Y[1]
            prey = Y[2]
            
            # check for extinction
            if pred == 0 && prey == 0
                break
            end

            rates[1] = d1 * pred
            rates[2] = 2*b*prey*(800-(pred+prey)) / 800
            rates[3] = 2*p2*pred*prey/800 +d2*prey
            rates[4] = 2*p1*pred*prey/800
            tot_rate = sum(rates)
            #println(pred, " ", prey, " ", rates[1], " ", rates[2], " ", rates[3], " ", rates[4])

            tau = rand(Exponential(1 / tot_rate))

            if t + tau > 2
                break
            end

            t += tau

            u = rand() * tot_rate
            if rates[1] > u
                Y[1] -= 1
            elseif (rates[1] + rates[2]) > u
                Y[2] += 1
            elseif (rates[1] + rates[2] + rates[3]) > u
                Y[2] -= 1
            else
                Y[1] += 1
                Y[2] -= 1
            end
        end
        state[i, :] .= Y
        # log prob of the observation given the state under the binomial error model
        if state[i, 1] < obs[1] || state[i, 2] < obs[2]
            log_weights[i] = -Inf
        else
            log_weights[i] = logpdf(Binomial(state[i, 1], q), obs[1]) + logpdf(Binomial(state[i, 2], q), obs[2])
        end
    end
end

function PP_init_state(init_obs, q, num_particles)
    init_state = zeros(num_particles,2)
    proposed_pred = 0
    proposed_prey = 0
    for i in 1:num_particles
        while true
            proposed_pred = rand(NegativeBinomial(init_obs[1] + 1, q)) + init_obs[1]
            proposed_prey = rand(NegativeBinomial(init_obs[2] + 1, q)) + init_obs[2]
            if proposed_pred + proposed_prey <= 800
                break
            end
        end
        init_state[i,1] = proposed_pred
        init_state[i,2] = proposed_prey
    end
    return init_state
end


function PP_likelihood_bootstrap(b, d1, d2, p1, p2, obs, obs_err, num_particles)

    l = size(obs)[2]

    log_weights = zeros(num_particles)
    indices = zeros(Int, num_particles) # preallocate for systematic resampling
    normalized_weights = zeros(num_particles)
    sorted_uniforms = zeros(num_particles)

    LL = zeros(l-1)

    @views state = PP_init_state(obs[:,1], obs_err, num_particles)

    for j in 1:(l-1)
        @views PP_forward!(b, d1, d2, p1, p2, state, log_weights, obs[:,j+1], obs_err)
        log_sum_exp_weights = log_sum_exp(log_weights)
        LL[j] = log_sum_exp_weights - log(num_particles)
        normalized_weights .= exp.(log_weights .- log_sum_exp_weights)
        systematic_resample!(normalized_weights, state, indices, sorted_uniforms)
    end

    return sum(LL)
end



function main(obs_err, n_particles, num_mcmc_samples; C=nothing)
    @assert obs_err in [0.5, 0.7, 0.9]
    if obs_err == 0.5
        data = readdlm("PP_05.csv", ',', Int64) # how to read the data in
    elseif obs_err == 0.7
        data = readdlm("PP_07.csv", ',', Int64) 
    else
        data = readdlm("PP_09.csv", ',', Int64) 
    end

    function create_density(data, num_particles)
        function density(theta)
            if theta[1] > 0 # log b
                return -Inf
            elseif theta[2] > 0 # log d1
                return -Inf
            elseif theta[3] > 0 # log d2
                return -Inf 
            elseif theta[4] < 0.01 # p1
                return -Inf
            elseif theta[5] < 0 # p2
                return -Inf
            else
                b = exp(theta[1])
                d1 = exp(theta[2])
                d2 = exp(theta[3])
                p1 = theta[4]
                p2 = theta[5]
                return PP_likelihood_bootstrap(b, d1, d2, p1, p2, data, obs_err, num_particles) + 
                        logpdf(Normal(log(0.25), 0.25), theta[1]) +
                        logpdf(Normal(log(0.1), 0.5), theta[2]) +
                        logpdf(Normal(log(0.01), 0.5), theta[3]) +
                        logpdf(Exponential(0.1), p1) +
                        logpdf(Exponential(0.05), p2)
            end
        end
        return density
    end
    density = create_density(data, n_particles)

    if isnothing(C)
        C = [0.00961312 0.00151433 0.00556913 0.00021355 0.00162792;
            0.00151433 0.02050758 0.003921 0.00274817 -0.00253268;
            0.00556913 0.003921 0.2050535 0.0005125 -0.00288658;
            0.00021355 0.00274817 0.0005125 0.00037904 -0.00034327;
            0.00162792 -0.00253268 -0.00288658 -0.00034327 0.00074064]
    end
    rw_prop = RWMH(MvNormal(0.5 * C))
    model = DensityModel(density)
    chain = sample(model, rw_prop, num_mcmc_samples; param_names=["log b", "log d1", "log d2", "p1", "p2"], chain_type=Chains, initial_params=[log(0.26), log(0.1), log(0.01), 0.13, 0.05])
    return chain
end