using DelimitedFiles
using DataStructures
using LinearAlgebra
using Random, Distributions, AdvancedMH, MCMCChains
using Plots, StatsBase, StatsPlots
include("../ParticleFilter.jl")


function forward_day_is!(N, beta_p, beta_s, sigma, gamma, q, state, weights, num_infections, NF; last_day=false)

    num_particles = size(state, 1)

    Threads.@threads for i in 1:num_particles

        @views Z = state[i, :]
        event_stack = Stack{Int64}()
        time_stack = Stack{Float64}()
        t = 0
        a = zeros(5)
        b = zeros(5)

        infection_times = sort!(rand(num_infections))
        w = -sum(log.(1:num_infections))

        for j in num_infections:-1:1
            push!(event_stack, 3)
            push!(time_stack, infection_times[j])
        end

        while !isempty(event_stack)
            S = N - Z[1]
            E = Z[1] - Z[2] - Z[5]
            Ip = Z[2] - Z[3]
            Is = Z[3] - Z[4]

            a[1] = S * (beta_p * Ip + beta_s * Is) / N
            a[2] = q * sigma * E
            a[3] = gamma * Ip
            a[4] = gamma * Is
            a[5] = (1 - q) * sigma * E
            a0 = sum(a)

            if top(event_stack) == 3
                if Ip == 0
                    forced_event = (E == 0) ? 1 : 2
                else
                    forced_event = 0
                end
            elseif top(event_stack) == 2
                forced_event = (E == 0) ? 1 : 0
            else
                forced_event = 0
            end

            if forced_event > 0
                lambda = a[forced_event]
                t_next = top(time_stack)
                t_dash = rand(Truncated(Exponential(1 / lambda), 0, t_next - t))
                w += -log(lambda) + lambda * t_dash + log(1 - exp(-lambda * (t_next - t)))
                push!(event_stack, forced_event)
                push!(time_stack, t + t_dash)
            end

            b[1] = (top(event_stack) == 1 || (Z[1] == NF && q == 1)) ? 0 : a[1]
            b[2] = (top(event_stack) == 2 || Z[2] == NF) ? 0 : a[2]
            b[3] = 0
            b[4] = (Z[1] - Z[4] - Z[5] == 1) ? 0 : a[4]
            b[5] = ((Z[5] == N - NF) || (Z[1] - Z[4] - Z[5] == 1)) ? 0 : a[5]
            b0 = sum(b)

            t_dash = rand(Exponential(1 / b0))

            if t + t_dash < top(time_stack)
                event = rand(Categorical(b / b0))
                Z[event] += 1
                t += t_dash
                w += (log(a[event]) - log(b[event]) - (a0 - b0) * t_dash)
            else
                event = pop!(event_stack)
                t_new = pop!(time_stack)
                Z[event] += 1
                w += (log(a[event]) - (a0 - b0) * (t_new - t))
                t = t_new
            end
        end

        while true

            S = N - Z[1]
            E = Z[1] - Z[2] - Z[5]
            Ip = Z[2] - Z[3]
            Is = Z[3] - Z[4]

            a[1] = S * (beta_p * Ip + beta_s * Is) / N
            a[2] = q * sigma * E
            a[3] = gamma * Ip
            a[4] = gamma * Is
            a[5] = (1 - q) * sigma * E
            a0 = sum(a)

            b[1] = (Z[1] == NF && q == 1) ? 0 : a[1]
            b[2] = (Z[2] == NF) ? 0 : a[2]
            b[3] = 0
            b[4] = ((Z[1] - Z[4] - Z[5] == 1) && !last_day) ? 0 : a[4]
            b[5] = ((Z[5] == N - NF) || ((Z[1] - Z[4] - Z[5] == 1) && !last_day)) ? 0 : a[5]
            b0 = sum(b)

            t_dash = rand(Exponential(1 / b0))

            if t + t_dash < 1
                event = rand(Categorical(b / b0))
                Z[event] += 1
                t += t_dash
                w += (log(a[event]) - log(b[event]) - (a0 - b0) * t_dash)
            else
                w += -(a0 - b0) * (1 - t)
                break
            end
        end
        weights[i] = w
        state[i, :] = Z
    end
end

function forward_end!(N, beta_p, beta_s, sigma, gamma, q, state, weights, NF)
    num_particles = size(state, 1)
    a = zeros(5)
    b = zeros(5)

    for i in 1:num_particles
        @views Z = state[i, :]
        w = 0

        while true

            S = N - Z[1]
            E = Z[1] - Z[2] - Z[5]
            Ip = Z[2] - Z[3]
            Is = Z[3] - Z[4]

            if (E + Ip + Is) == 0
                break
            end

            a[1] = S * (beta_p * Ip + beta_s * Is) / N
            a[2] = q * sigma * E
            a[3] = gamma * Ip
            a[4] = gamma * Is
            a[5] = (1 - q) * sigma * E
            a0 = sum(a)

            b[1] = (Z[1] == NF && q == 1) ? 0 : a[1]
            b[2] = 0
            b[3] = 0
            b[4] = a[4]
            b[5] = (Z[5] == N - NF) ? 0 : a[5]
            b0 = sum(b)

            t_dash = rand(Exponential(1 / b0))
            event = rand(Categorical(b / b0))
            Z[event] += 1
            w += (log(a[event]) - log(b[event]) - (a0 - b0) * t_dash)
        end
        state[i, :] = Z
        weights[i] = w
    end
end


function SEIAR_init_state(N, beta_p, beta_s, sigma, gamma, q, num_particles, NF, weights)
    state = zeros(Int, num_particles, 5)
    rates = zeros(5)
    for i in 1:num_particles
        state[i, 1] = 1;
        state[i, 2] = 1;
        while true
            S = N - state[i, 1]
            E = state[i, 1] - state[i, 2] - state[i, 5]
            Ip = state[i, 2] - state[i, 3]
            Is = state[i, 3] - state[i, 4]

            rates[1] = S * (beta_p * Ip + beta_s * Is) / N
            rates[2] = q * sigma * E
            rates[3] = gamma * Ip
            rates[4] = gamma * Is
            rates[5] = (1 - q) * sigma * E
            total_rate = sum(rates)

            idx = rand(Categorical(rates / total_rate))
            #if state[i, 2] == 0 && idx == 5
            #    continue
            #end

            state[i, idx] += 1
            if idx == 3 # We now have one symptomatic individual, so this is our initial state
                break
            end
        end
        # required for consistency
        weights[i] = (state[i, 2] <= NF) && (state[i, 5] <= (N - NF)) ? 0 : -Inf
    end
    return state
end



function SEIAR_likelihood_is(N, beta_p, beta_s, sigma, gamma, q, y, num_particles)

    l = length(y)

    log_weights = zeros(num_particles)
    indices = zeros(Int, num_particles) # preallocate for systematic resampling
    normalized_weights = zeros(num_particles)
    sorted_uniforms = zeros(num_particles)

    LL = zeros(l + 1)

    NF = cumsum(y)[end] + 1
    state = SEIAR_init_state(N, beta_p, beta_s, sigma, gamma, q, num_particles, NF, log_weights)

    if any(log_weights .== -Inf)
        log_sum_exp_weights = log_sum_exp(log_weights)
        LL_init = log_sum_exp_weights - log(num_particles)
        normalized_weights .= exp.(log_weights .- log_sum_exp_weights)
        systematic_resample!(normalized_weights, state, indices, sorted_uniforms)
    else
        LL_init = 0
    end

    for j in 1:(l-1)
        forward_day_is!(N, beta_p, beta_s, sigma, gamma, q, state, log_weights, y[j], NF)
        log_sum_exp_weights = log_sum_exp(log_weights)
        LL[j] = log_sum_exp_weights - log(num_particles)
        normalized_weights .= exp.(log_weights .- log_sum_exp_weights)
        systematic_resample!(normalized_weights, state, indices, sorted_uniforms)
    end

    forward_day_is!(N, beta_p, beta_s, sigma, gamma, q, state, log_weights, y[end], NF; last_day=true)
    log_sum_exp_weights = log_sum_exp(log_weights)
    LL[l] = log_sum_exp_weights - log(num_particles)
    normalized_weights .= exp.(log_weights .- log_sum_exp_weights)
    systematic_resample!(normalized_weights, state, indices, sorted_uniforms)

    forward_end!(N, beta_p, beta_s, sigma, gamma, q, state, log_weights, NF)
    LL[l + 1] = log_sum_exp(log_weights) - log(num_particles)

    return sum(LL) + LL_init + log(q) # log q because we didn't let the intial guy become asymptomatic
end



function main(pop_size, n_particles, num_mcmc_samples; C=nothing)
    @assert pop_size in [150, 350, 500, 1000, 2000]
    data = readdlm("seiar$pop_size.csv", ',', Int64) # how to read the data in
    daily_counts = data[2:end] - data[1:(end-1)]

    function create_density(daily_counts, n_particles)
        function density(theta)
            prior = Gamma(10, 1 / 10)
            if theta[1] < 0.1 || theta[1] > 8
                return -Inf
            elseif theta[2] < 0.1
                return -Inf
            elseif theta[3] < 0.5
                return -Inf 
            elseif theta[4] < 0 || theta[4] > 1
                return -Inf
            elseif theta[5] < 0.5 || theta[5] > 1
                return -Inf
            else
                sigma = 1 / theta[2]
                gamma = 1 / theta[3]
                q = theta[5]
                beta_p = gamma * theta[4] * theta[1] / q
                beta_s = gamma * (1 - theta[4]) * theta[1] / q
                return SEIAR_likelihood_is(pop_size, beta_p, beta_s, sigma, gamma, q, daily_counts, n_particles) + 
                        logpdf(prior, theta[2]) +
                        logpdf(prior, theta[3])
            end
        end
        return density
    end
    density = create_density(daily_counts, n_particles)

    if isnothing(C)
        if pop_size == 150
        # run w 80 particles, 3 * 10^5 mcmc steps
            C = [0.249869    0.0313492    0.0349223   -0.0251837   -0.0160679;
                 0.0313492   0.0982451   -0.0292783    0.00659737  -0.00347467;
                 0.0349223  -0.0292783    0.0727815    0.0166649   -0.00262974;
                -0.0251837   0.00659737   0.0166649    0.0577057    0.00259436;
                -0.0160679  -0.00347467  -0.00262974   0.00259436   0.00335701]
        elseif pop_size == 350
            C = [0.0658676    0.0143006    0.023272    -0.00475914  -0.0126936;
                0.0143006    0.103558    -0.0310766    0.0168814   -0.00409237;
                0.023272    -0.0310766    0.0939461    0.0155036   -0.00549904;
                -0.00475914   0.0168814    0.0155036    0.0541026    0.00148719;
                -0.0126936   -0.00409237  -0.00549904   0.00148719   0.00566848]
        elseif pop_size == 500
            C = [0.0490791   0.0159622    0.0196186   -0.0098734   -0.0107154;
                 0.0159622   0.0769658   -0.0129195    0.0230084   -0.00416212;
                 0.0196186  -0.0129195    0.0664715    0.0220156   -0.00587622;
                -0.0098734   0.0230084    0.0220156    0.0787558    0.00244592;
                -0.0107154  -0.00416212  -0.00587622   0.00244592   0.00476275]
        elseif pop_size == 1000
            C = [0.024747     0.0100976    0.0115773   -0.00204267   -0.00865539;
                 0.0100976    0.0604175   -0.00590266   0.0207981    -0.00512788;
                 0.0115773   -0.00590266   0.0573138    0.0259788    -0.00551361;
                -0.00204267   0.0207981    0.0259788    0.072653      0.000803354;
                -0.00865539  -0.00512788  -0.00551361   0.000803354   0.00524213]
        else
            C = [0.0233245    0.01301      0.0160311   -0.00499184  -0.0071838;
  0.01301      0.0653933   -0.0176995    0.00924847  -0.00553246;
  0.0160311   -0.0176995    0.0559406    0.00916293  -0.00484323;
 -0.00499184   0.00924847   0.00916293   0.0519987    0.002257;
 -0.0071838   -0.00553246  -0.00484323   0.002257     0.00311867]
        end
    end
    rw_prop = RWMH(MvNormal(0.5 * C))
    model = DensityModel(density)
    chain = sample(model, rw_prop, num_mcmc_samples; param_names=["R0", "1/\\sigma", "1/\\gamma", "\\kappa", "q"], chain_type=Chains, initial_params=[2.2, 1., 1., 0.7, 0.9])
    return chain
end