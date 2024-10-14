import numpy as np
import numba


@numba.jit(nopython=True, fastmath=True)
def SIR(N, beta, gamma, initial_infected, time_run, remove_trivial=False):
    nsims = beta.shape[0]
    Z1_PO = np.zeros((nsims, time_run, 1))
    Z2_PO = np.zeros((nsims, time_run, 1))

    Z1_PO[:, 0, 0] = initial_infected

    for i in range(nsims):
        beta_ = beta[i]
        gamma_ = gamma[i]
        N_ = N[i]

        Z1 = initial_infected
        Z2 = 0

        cum_time = 0
        while True:
            rates = np.array(
                [
                    beta_ * (N_ - Z1) * (Z1 - Z2) / (N_ - initial_infected),
                    gamma_ * (Z1 - Z2),
                ]
            )
            total_rate = np.sum(rates)
            t_new = np.random.exponential(1 / total_rate)
            time_steps_traversed = int(np.floor(cum_time + t_new) - np.floor(cum_time))
            if time_steps_traversed > 0:
                Z1_PO[
                    i,
                    int(np.floor(cum_time) + 1) : int(
                        np.floor(cum_time) + time_steps_traversed + 1
                    ),
                    0,
                ] = Z1
                Z2_PO[
                    i,
                    int(np.floor(cum_time) + 1) : int(
                        np.floor(cum_time) + time_steps_traversed + 1
                    ),
                    0,
                ] = Z2

            cum_time += t_new
            if cum_time > time_run:
                break

            probs = rates / total_rate
            cdf = np.cumsum(probs)
            u = np.random.rand()

            if u < cdf[0]:
                Z1 += 1
            else:
                Z2 += 1

            if (Z1 - Z2) == 0:
                if remove_trivial:
                    if Z1 == initial_infected:
                        Z1_PO[i, :, 0] = 0
                        Z2_PO[i, :, 0] = 0

                        Z1_PO[i, 0, 0] = initial_infected

                        Z1 = initial_infected
                        Z2 = 0

                        cum_time = 0
                        continue
                Z1_PO[i, int(np.ceil(cum_time)) :, 0] = Z1
                Z2_PO[i, int(np.ceil(cum_time)) :, 0] = Z2
                break
    return Z1_PO, Z2_PO


@numba.jit(nopython=True, fastmath=True)
def SIIR(N, beta, gamma, initial_infected, time_run, remove_trivial=False):
    nsims = beta.shape[0]
    Z1_PO = np.zeros((nsims, time_run, 1))  # Infectious stage 1
    Z2_PO = np.zeros((nsims, time_run, 1))  # Infectious stage 2
    Z3_PO = np.zeros((nsims, time_run, 1))

    Z1_PO[:, 0, 0] = initial_infected

    for i in range(nsims):
        beta_ = beta[i]
        gamma_ = gamma[i]
        N_ = N[i]

        Z1 = initial_infected
        Z2 = 0
        Z3 = 0

        cum_time = 0
        while True:
            rates = np.array(
                [
                    beta_ * (N_ - Z1) * (Z1 - Z3) / (N_ - initial_infected),
                    2 * gamma_ * (Z1 - Z2),
                    2 * gamma_ * (Z2 - Z3),
                ]
            )
            total_rate = np.sum(rates)
            t_new = np.random.exponential(1 / total_rate)
            time_steps_traversed = int(np.floor(cum_time + t_new) - np.floor(cum_time))
            if time_steps_traversed > 0:
                Z1_PO[
                    i,
                    int(np.floor(cum_time) + 1) : int(
                        np.floor(cum_time) + time_steps_traversed + 1
                    ),
                    0,
                ] = Z1
                Z2_PO[
                    i,
                    int(np.floor(cum_time) + 1) : int(
                        np.floor(cum_time) + time_steps_traversed + 1
                    ),
                    0,
                ] = Z2
                Z3_PO[
                    i,
                    int(np.floor(cum_time) + 1) : int(
                        np.floor(cum_time) + time_steps_traversed + 1
                    ),
                    0,
                ] = Z3

            cum_time += t_new
            if cum_time > time_run:
                break

            probs = rates / total_rate
            cdf = np.cumsum(probs)
            u = np.random.rand()

            if u < cdf[0]:
                Z1 += 1
            elif u < cdf[1]:
                Z2 += 1
            else:
                Z3 += 1

            if (Z1 - Z3) == 0:
                if remove_trivial:
                    if Z1 == initial_infected:
                        Z1_PO[i, :, 0] = 0
                        Z2_PO[i, :, 0] = 0
                        Z3_PO[i, :, 0] = 0

                        Z1_PO[i, 0, 0] = initial_infected

                        Z1 = initial_infected
                        Z2 = Z3 = 0

                        cum_time = 0
                        continue
                Z1_PO[i, int(np.ceil(cum_time)) :, 0] = Z1
                Z2_PO[i, int(np.ceil(cum_time)) :, 0] = Z2
                Z3_PO[i, int(np.ceil(cum_time)) :, 0] = Z3
                break
    return Z1_PO, Z2_PO, Z3_PO


@numba.jit(nopython=True, fastmath=True)
def SI10R(N, beta, gamma, initial_infected, time_run, remove_trivial=False):
    nsims = beta.shape[0]
    Z1_PO = np.zeros((nsims, time_run, 1))  # Infectious stage 1
    Z2_PO = np.zeros((nsims, time_run, 1))  # Infectious stage 2
    Z3_PO = np.zeros((nsims, time_run, 1))  # Infectious stage 3
    Z4_PO = np.zeros((nsims, time_run, 1))
    Z5_PO = np.zeros((nsims, time_run, 1))
    Z6_PO = np.zeros((nsims, time_run, 1))
    Z7_PO = np.zeros((nsims, time_run, 1))
    Z8_PO = np.zeros((nsims, time_run, 1))
    Z9_PO = np.zeros((nsims, time_run, 1))
    Z10_PO = np.zeros((nsims, time_run, 1))
    Z11_PO = np.zeros((nsims, time_run, 1))

    Z1_PO[:, 0, 0] = initial_infected

    for i in range(nsims):
        beta_ = beta[i]
        gamma_ = gamma[i]
        N_ = N[i]

        Z1 = initial_infected
        Z2 = 0
        Z3 = 0
        Z4 = 0
        Z5 = 0
        Z6 = 0
        Z7 = 0
        Z8 = 0
        Z9 = 0
        Z10 = 0
        Z11 = 0

        cum_time = 0
        while True:
            rates = np.array(
                [
                    beta_ * (N_ - Z1) * (Z1 - Z11) / (N_ - initial_infected),
                    10 * gamma_ * (Z1 - Z2),
                    10 * gamma_ * (Z2 - Z3),
                    10 * gamma_ * (Z3 - Z4),
                    10 * gamma_ * (Z4 - Z5),
                    10 * gamma_ * (Z5 - Z6),
                    10 * gamma_ * (Z6 - Z7),
                    10 * gamma_ * (Z7 - Z8),
                    10 * gamma_ * (Z8 - Z9),
                    10 * gamma_ * (Z9 - Z10),
                    10 * gamma_ * (Z10 - Z11),
                ]
            )
            total_rate = np.sum(rates)
            t_new = np.random.exponential(1 / total_rate)
            time_steps_traversed = int(np.floor(cum_time + t_new) - np.floor(cum_time))
            if time_steps_traversed > 0:
                Z1_PO[
                    i,
                    int(np.floor(cum_time) + 1) : int(
                        np.floor(cum_time) + time_steps_traversed + 1
                    ),
                    0,
                ] = Z1
                Z2_PO[
                    i,
                    int(np.floor(cum_time) + 1) : int(
                        np.floor(cum_time) + time_steps_traversed + 1
                    ),
                    0,
                ] = Z2
                Z3_PO[
                    i,
                    int(np.floor(cum_time) + 1) : int(
                        np.floor(cum_time) + time_steps_traversed + 1
                    ),
                    0,
                ] = Z3
                Z4_PO[
                    i,
                    int(np.floor(cum_time) + 1) : int(
                        np.floor(cum_time) + time_steps_traversed + 1
                    ),
                    0,
                ] = Z4
                Z5_PO[
                    i,
                    int(np.floor(cum_time) + 1) : int(
                        np.floor(cum_time) + time_steps_traversed + 1
                    ),
                    0,
                ] = Z5
                Z6_PO[
                    i,
                    int(np.floor(cum_time) + 1) : int(
                        np.floor(cum_time) + time_steps_traversed + 1
                    ),
                    0,
                ] = Z6
                Z7_PO[
                    i,
                    int(np.floor(cum_time) + 1) : int(
                        np.floor(cum_time) + time_steps_traversed + 1
                    ),
                    0,
                ] = Z7
                Z8_PO[
                    i,
                    int(np.floor(cum_time) + 1) : int(
                        np.floor(cum_time) + time_steps_traversed + 1
                    ),
                    0,
                ] = Z8
                Z9_PO[
                    i,
                    int(np.floor(cum_time) + 1) : int(
                        np.floor(cum_time) + time_steps_traversed + 1
                    ),
                    0,
                ] = Z9
                Z10_PO[
                    i,
                    int(np.floor(cum_time) + 1) : int(
                        np.floor(cum_time) + time_steps_traversed + 1
                    ),
                    0,
                ] = Z10
                Z11_PO[
                    i,
                    int(np.floor(cum_time) + 1) : int(
                        np.floor(cum_time) + time_steps_traversed + 1
                    ),
                    0,
                ] = Z11

            cum_time += t_new
            if cum_time > time_run:
                break

            probs = rates / total_rate
            cdf = np.cumsum(probs)
            u = np.random.rand()

            if u < cdf[0]:
                Z1 += 1
            elif u < cdf[1]:
                Z2 += 1
            elif u < cdf[2]:
                Z3 += 1
            elif u < cdf[3]:
                Z4 += 1
            elif u < cdf[4]:
                Z5 += 1
            elif u < cdf[5]:
                Z6 += 1
            elif u < cdf[6]:
                Z7 += 1
            elif u < cdf[7]:
                Z8 += 1
            elif u < cdf[8]:
                Z9 += 1
            elif u < cdf[9]:
                Z10 += 1
            else:
                Z11 += 1

            if (Z1 - Z11) == 0:
                if remove_trivial:
                    if Z1 == initial_infected:
                        Z1_PO[i, :, 0] = 0
                        Z2_PO[i, :, 0] = 0
                        Z3_PO[i, :, 0] = 0
                        Z4_PO[i, :, 0] = 0
                        Z5_PO[i, :, 0] = 0
                        Z6_PO[i, :, 0] = 0
                        Z7_PO[i, :, 0] = 0
                        Z8_PO[i, :, 0] = 0
                        Z9_PO[i, :, 0] = 0
                        Z10_PO[i, :, 0] = 0
                        Z11_PO[i, :, 0] = 0

                        Z1_PO[i, 0, 0] = initial_infected

                        Z1 = initial_infected
                        Z2 = Z3 = Z4 = Z5 = Z6 = Z7 = Z8 = Z9 = Z10 = Z11 = 0

                        cum_time = 0
                        continue
                Z1_PO[i, int(np.ceil(cum_time)) :, 0] = Z1
                Z2_PO[i, int(np.ceil(cum_time)) :, 0] = Z2
                Z3_PO[i, int(np.ceil(cum_time)) :, 0] = Z3
                Z4_PO[i, int(np.ceil(cum_time)) :, 0] = Z4
                Z5_PO[i, int(np.ceil(cum_time)) :, 0] = Z5
                Z6_PO[i, int(np.ceil(cum_time)) :, 0] = Z6
                Z7_PO[i, int(np.ceil(cum_time)) :, 0] = Z7
                Z8_PO[i, int(np.ceil(cum_time)) :, 0] = Z8
                Z9_PO[i, int(np.ceil(cum_time)) :, 0] = Z9
                Z10_PO[i, int(np.ceil(cum_time)) :, 0] = Z10
                Z11_PO[i, int(np.ceil(cum_time)) :, 0] = Z11
                break
    return Z1_PO, Z2_PO, Z3_PO, Z4_PO, Z5_PO, Z6_PO, Z7_PO, Z8_PO, Z9_PO, Z10_PO, Z11_PO


@numba.jit(nopython=True, fastmath=True)
def SIR_final_size(N, beta, gamma, initial_infected, max_len=100):
    Z1_PO, Z2_PO = SIR(N, beta, gamma, initial_infected, time_run=max_len)
    I = Z1_PO - Z2_PO
    if np.all(I[:, -1, 0] == 0):
        longest_obs = 1
        for i in range(I.shape[0]):
            die_out = np.where(I[i, :, 0] == 0)[0][0]
            if die_out > longest_obs:
                longest_obs = die_out
    else:
        longest_obs = max_len
    return Z1_PO[:, : (longest_obs + 1), :], Z2_PO[:, : (longest_obs + 1), :]


@numba.jit(nopython=True, fastmath=True)
def SIIR_final_size(N, beta, gamma, initial_infected, max_len=100):
    Z1_PO, Z2_PO, Z3_PO = SIIR(N, beta, gamma, initial_infected, time_run=max_len)
    I = Z1_PO - Z3_PO
    if np.all(I[:, -1, 0] == 0):
        longest_obs = 1
        for i in range(I.shape[0]):
            die_out = np.where(I[i, :, 0] == 0)[0][0]
            if die_out > longest_obs:
                longest_obs = die_out
    else:
        longest_obs = max_len
    return (
        Z1_PO[:, : (longest_obs + 1), :],
        Z2_PO[:, : (longest_obs + 1), :],
        Z3_PO[:, : (longest_obs + 1), :],
    )


@numba.jit(nopython=True, fastmath=True)
def SEIAR_init_state(N, beta_p, beta_s, sigma, gamma, q, initial_infected):
    state = np.zeros(5)
    state[0] = 1
    state[1] = 1  # one initially pre-symptomatic
    while True:
        rates = np.array(
            [
                (N - state[0])
                * (beta_p * (state[1] - state[2]) + beta_s * (state[2] - state[3]))
                / (N - initial_infected),
                q * sigma * (state[0] - state[1] - state[4]),
                gamma * (state[1] - state[2]),
                gamma * (state[2] - state[3]),
                (1 - q) * sigma * (state[0] - state[1] - state[4]),
            ]
        )
        probs = rates / np.sum(rates)
        cdf = np.cumsum(probs)
        u = np.random.rand()

        if u < cdf[0]:
            state[0] += 1
        elif u < cdf[1]:
            state[1] += 1
        elif u < cdf[2]:
            state[2] += 1
            break  # we now have one symptomatic infection and therefore our initial state
        elif u < cdf[3]:
            state[3] += 1
        else:
            state[4] += 1
    return state


@numba.jit(nopython=True, fastmath=True)
def SEIAR(N, beta_p, beta_s, sigma, gamma, q, initial_infected, time_run):
    nsims = beta_p.shape[0]
    Z1_PO = np.zeros((nsims, time_run, 1))
    Z2_PO = np.zeros((nsims, time_run, 1))
    Z3_PO = np.zeros((nsims, time_run, 1))
    Z4_PO = np.zeros((nsims, time_run, 1))
    Z5_PO = np.zeros((nsims, time_run, 1))

    for i in range(nsims):
        beta_p_ = beta_p[i]
        beta_s_ = beta_s[i]
        sigma_ = sigma[i]
        gamma_ = gamma[i]
        q_ = q[i]

        initial_state = SEIAR_init_state(
            N, beta_p_, beta_s_, sigma_, gamma_, q_, initial_infected
        )

        Z1_PO[i, 0, 0] = Z1 = initial_state[0]
        Z2_PO[i, 0, 0] = Z2 = initial_state[1]
        Z3_PO[i, 0, 0] = Z3 = initial_infected  # this will always be 1 by construction
        # will always be 0 by construction (no removals until after there is one symptomatic individual)
        Z4 = 0
        # may be non-zero as exposures may become asymptomatic before any symptomatic individuals show up
        Z5_PO[i, 0, 0] = Z5 = initial_state[4]
        cum_time = 0
        while True:
            rates = np.array(
                [
                    (N - Z1)
                    * (beta_p_ * (Z2 - Z3) + beta_s_ * (Z3 - Z4))
                    / (N - initial_infected),
                    q_ * sigma_ * (Z1 - Z2 - Z5),
                    gamma_ * (Z2 - Z3),
                    gamma_ * (Z3 - Z4),
                    (1 - q_) * sigma_ * (Z1 - Z2 - Z5),
                ]
            )
            total_rate = np.sum(rates)
            t_new = np.random.exponential(1 / total_rate)
            time_steps_traversed = int(np.floor(cum_time + t_new) - np.floor(cum_time))
            if time_steps_traversed > 0:
                # Record partial observations if time step is traversed
                Z1_PO[
                    i,
                    int(np.floor(cum_time) + 1) : int(
                        np.floor(cum_time) + time_steps_traversed + 1
                    ),
                    0,
                ] = Z1
                Z2_PO[
                    i,
                    int(np.floor(cum_time) + 1) : int(
                        np.floor(cum_time) + time_steps_traversed + 1
                    ),
                    0,
                ] = Z2
                Z3_PO[
                    i,
                    int(np.floor(cum_time) + 1) : int(
                        np.floor(cum_time) + time_steps_traversed + 1
                    ),
                    0,
                ] = Z3
                Z4_PO[
                    i,
                    int(np.floor(cum_time) + 1) : int(
                        np.floor(cum_time) + time_steps_traversed + 1
                    ),
                    0,
                ] = Z4
                Z5_PO[
                    i,
                    int(np.floor(cum_time) + 1) : int(
                        np.floor(cum_time) + time_steps_traversed + 1
                    ),
                    0,
                ] = Z5

            cum_time += t_new
            if cum_time > time_run:
                break

            probs = rates / total_rate
            cdf = np.cumsum(probs)
            u = np.random.rand()

            if u < cdf[0]:
                Z1 += 1
            elif u < cdf[1]:
                Z2 += 1
            elif u < cdf[2]:
                Z3 += 1
            elif u < cdf[3]:
                Z4 += 1
            else:
                Z5 += 1

            if (Z2 == Z4) and (Z1 == Z2 + Z5):
                Z1_PO[i, int(np.ceil(cum_time)) :, 0] = Z1
                Z2_PO[i, int(np.ceil(cum_time)) :, 0] = Z2
                Z3_PO[i, int(np.ceil(cum_time)) :, 0] = Z3
                Z4_PO[i, int(np.ceil(cum_time)) :, 0] = Z4
                Z5_PO[i, int(np.ceil(cum_time)) :, 0] = Z5
                break
    return Z1_PO, Z2_PO, Z3_PO, Z4_PO, Z5_PO


@numba.jit(nopython=True, fastmath=True)
def PP(
    b,
    d1,
    d2,
    p1,
    p2,
    carrying_capacity,
    initial_pred,
    initial_prey,
    time_run,
):
    nsims = b.shape[0]
    pred_PO = np.zeros((nsims, time_run, 1))
    prey_PO = np.zeros((nsims, time_run, 1))

    for i in range(nsims):
        b_ = b[i]
        d1_ = d1[i]
        d2_ = d2[i]
        p1_ = p1[i]
        p2_ = p2[i]
        pred_PO[i, 0, 0] = initial_pred[i]
        prey_PO[i, 0, 0] = initial_prey[i]
        # stop_flag = False

        pred, prey = initial_pred[i], initial_prey[i]
        cum_time = 0
        while True:
            r1 = d1_ * pred  # (pred,prey) -> (pred-1,prey)
            r2 = (
                2 * b_ * prey * (carrying_capacity - pred - prey) / carrying_capacity
            )  # (pred,prey) -> (pred,prey+1)
            r3 = (
                2 * p2_ * pred * prey / carrying_capacity + d2_ * prey
            )  # (pred,prey) -> (pred,prey-1)
            r4 = (
                2 * p1_ * pred * prey / carrying_capacity
            )  # (pred,prey) -> (pred+1,prey-1)
            rates = np.array([r1, r2, r3, r4])
            total_rate = np.sum(rates)
            t_new = np.random.exponential(1 / total_rate)
            # Number of integer time steps traversed
            time_steps_traversed = int(np.floor(cum_time + t_new) - np.floor(cum_time))
            if time_steps_traversed > 0:
                # Record partial observations if time step is traversed
                pred_PO[
                    i,
                    int(np.floor(cum_time) + 1) : int(
                        np.floor(cum_time) + time_steps_traversed + 1
                    ),
                    0,
                ] = pred
                prey_PO[
                    i,
                    int(np.floor(cum_time) + 1) : int(
                        np.floor(cum_time) + time_steps_traversed + 1
                    ),
                    0,
                ] = prey

                # if stop_flag:
                #    X_PO[i, int(np.ceil(cum_time) + 1) :, 0] = X
                #    Y_PO[i, int(np.ceil(cum_time) + 1) :, 0] = 0.0
                #    break

            cum_time = cum_time + t_new
            if cum_time > time_run:
                # Break the Gillespie algorithm
                break

            probs = rates / total_rate
            cdf = np.cumsum(probs)
            u = np.random.rand()

            if u < cdf[0]:
                pred -= 1
            elif u < cdf[1]:
                prey += 1
            elif u < cdf[2]:
                prey -= 1
            else:
                pred += 1
                prey -= 1

            # Extinction, all rates now 0
            if (pred + prey) == 0:
                pred_PO[i, int(np.ceil(cum_time)) :, 0] = pred
                prey_PO[i, int(np.ceil(cum_time)) :, 0] = prey
                break

            # if Y == 0 and X > max_num_prey:
            # Otherwise exponential growth occurs and simulations take forever
            #    stop_flag = True

    return pred_PO, prey_PO
