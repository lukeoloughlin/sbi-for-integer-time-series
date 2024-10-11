using DelimitedFiles
using Random, Distributions, AdvancedMH, MCMCChains
using Plots, StatsBase, StatsPlots
include("../ParticleFilter.jl")

"""
SIR forwards simulation using importance sampling
"""
function forward_day_is!(S, beta, gamma, X, Z1_cum, omega, NR)
    
    n_part = size(X, 1); # number of particles

    @Threads.threads for kk in 1:n_part

        a = zeros(2);
        Z = zeros(Int, 2);
        Z[1] = Z1_cum;
        Z[2] = X[kk];
        N = S;

        # generate the infection times
        tR = sort!(rand(NR));

        r = 1;
        L_imp = 0; #-gammaln(NR+1)
        t = 0;

        while r <= NR

            I = Z[1] - Z[2]

            a[1] = beta * (N - Z[1]) * I;
            a[2] = gamma * I;
            a0 = a[1] + a[2];

            if I > 1
                b = a[2];
            else
                b = 0.0;
            end

            t_dash = -log(rand()) / b;

                if t_dash + t < tR[r]
                    # recovery event

                    Z[2] = Z[2] + 1;
                    t = t + t_dash;

                    L_imp = L_imp + log(a[2]) - a0 * t_dash - (log(b) - b * t_dash);
                else

                    L_imp = L_imp + log(a[1]) - a0 * (tR[r] - t) + b * (tR[r] - t);

                    Z[1] = Z[1] + 1;
                    t = tR[r];
                    r = r + 1;
                end

        end

        # after the end of all the infection events then need to put in the last recoveries
        # in this part we can't let the process fade out

        while t < 1

            I = Z[1] - Z[2];
            a[1] = beta * (N - Z[1]) * I;
            a[2] = gamma * I;
            a0 = a[1] + a[2];

            if I > 1
                b = a[2];
            else
                b = 0;
            end

            t_dash = -log(rand()) / b;

            if t + t_dash > 1
                # gone over the next day

                L_imp = L_imp - a0 * (1 - t) + b * (1 - t);
                break;
            else
                # still within the day
                Z[2] = Z[2] + 1;
                t = t + t_dash;
                L_imp = L_imp + log(a[2]) - a0 * t_dash - (log(b) - b * t_dash);
            end

        end

        omega[kk] = L_imp
        X[kk] = Z[2];
    end
end

"""
SIR forwards simulation using importance sampling
"""

function forward_lastday_is!(S, beta, gamma, X, Z1_cum, omega, NR)

    n_part = size(X, 1); # number of rows

    Threads.@threads for kk in 1:n_part
        a = zeros(2);
        Z = zeros(Int, 2);
        Z[1] = Z1_cum
        Z[2] = X[kk]
        N = S

        # generate the infection times
        tR = sort!(rand(NR));

        r = 1;
        L_imp = 0 # -gammaln(NR+1)
        t = 0;

        while r <= NR

            I = Z[1] - Z[2];

            a[1] = beta * (N - Z[1]) * I;
            a[2] = gamma * I;
            a0 = a[1] + a[2];

            if I > 1
                b = a[2];
            else
                b = 0.0;
            end

            t_dash = -log(rand()) / b;

            if t_dash + t < tR[r]
                # recovery event

                Z[2] = Z[2] + 1;
                t = t + t_dash;

                L_imp = L_imp + log(a[2]) - a0 * t_dash - (log(b) - b * t_dash);
            else

                L_imp = L_imp + log(a[1]) - a0 * (tR[r] - t) + b * (tR[r] - t);

                Z[1] = Z[1] + 1;
                t = tR[r];
                r = r + 1;
            end
        end

        # as this is the last day, we can allow fadeout
        while t < 1

            I = Z[1] - Z[2];
            a[1] = beta * (N - Z[1]) * I;
            a[2] = gamma * I;
            a0 = a[1] + a[2];

            b = a[2]

            t_dash = -log(rand()) / b;

            if t + t_dash > 1
                # gone over the next day

                L_imp = L_imp - a0 * (1 - t) + b * (1 - t);
                break;
            else
                # still within the day
                Z[2] = Z[2] + 1;
                t = t + t_dash;

                L_imp = L_imp + log(a[2]) - a0 * t_dash - (log(b) - b * t_dash);
            end
        end

        omega[kk] = L_imp
        X[kk] = Z[2];
    end
end


"""
Return the probability of the epidemic fading out from the current state,
given no more infections occur
"""
function forward_end!(N, beta, gamma, X, Z1_final, omega)
    
    n_part = size(X, 1)
    p = gamma / (beta * (N - Z1_final) + gamma)

    # no more infections so Z1 does not change
    # which simplifies the calcuation

    Threads.@threads for kk in 1:n_part
        omega[kk] = (Z1_final - X[kk]) * log(p)
    end
end


function SIR_likelihood_is(N, beta_un, gamma, y, num_particles)

    if isnothing(y)
        l = 0
    else
        l = length(y)
    end

    Z2 = zeros(Int, num_particles)
    log_weights = zeros(num_particles)
    indices = zeros(Int, num_particles) # preallocate for systematic resampling
    normalized_weights = zeros(num_particles)
    sorted_uniforms = zeros(num_particles)

    # normalise here
    beta = beta_un / (N - 1);

    LL = zeros(l + 1);

    Z1 = 1

    for j in 1:(l-1)

        forward_day_is!(N, beta, gamma, Z2, Z1, log_weights, y[j])
        log_sum_exp_weights = log_sum_exp(log_weights)
        LL[j] = log_sum_exp_weights - log(num_particles)

        normalized_weights .= exp.(log_weights .- log_sum_exp_weights)
        Z1 = Z1 + y[j]
        #Z2 = systematic_resample(normalized_weights, Z2)
        systematic_resample!(normalized_weights, Z2, indices, sorted_uniforms)
    end

    if l > 0
        forward_lastday_is!(N, beta, gamma, Z2, Z1, log_weights, y[end])
        log_sum_exp_weights = log_sum_exp(log_weights)
        LL[l] = log_sum_exp_weights - log(num_particles)
        
        normalized_weights .= exp.(log_weights .- log_sum_exp_weights)
        Z1 = Z1 + y[end]
        #Z2 = systematic_resample(normalized_weights, Z2)
        Z2 = systematic_resample!(normalized_weights, Z2, indices)
    end

    # calculate the loglikelihood of no further infections
    forward_end!(N, beta, gamma, Z2, Z1, log_weights)
    LL[l + 1] = log_sum_exp(log_weights) - log(num_particles)

    return sum(LL)

end


function main(pop_size, n_particles, num_mcmc_samples)
    @assert pop_size in [50, 100, 200]
    data = readdlm("sir_$pop_size.csv", ',', Int64)[1, :] # how to read the data in
    daily_counts = data[2:end] - data[1:(end-1)]

    function create_density(daily_counts, n_particles)
        function density(theta)
            gamma_inv_prior = Gamma(10, 0.5)
            if theta[1] < 0 || theta[1] > 10
                return -Inf # Uniform prior on R0
            elseif theta[2] < 0
                return -Inf
            else
                return SIR_likelihood_is(pop_size, theta[1] / theta[2], 1 / theta[2], daily_counts, n_particles) + logpdf(gamma_inv_prior, theta[2])
            end
        end
        return density
    end
    density = create_density(daily_counts, n_particles)

    if pop_size == 50
        C = [0.362391  0.296886;
             0.296886  1.18312]
    elseif pop_size == 100
        C = [0.101385  0.123653;
             0.123653  0.929533]
    else
        C = [0.0678938  0.118561;
             0.118561   0.926801]
    end

    rw_prop = RWMH(MvNormal(0.5 * C))
    model = DensityModel(density)
    chain = sample(model, rw_prop, num_mcmc_samples; param_names=["R0", "1/\\gamma"], chain_type=Chains, initial_params=[2.0, 5.0])
    return chain
end