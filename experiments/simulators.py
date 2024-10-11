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
def SIIIR(N, beta, gamma, initial_infected, time_run, remove_trivial=False):
    nsims = beta.shape[0]
    Z1_PO = np.zeros((nsims, time_run, 1))  # Infectious stage 1
    Z2_PO = np.zeros((nsims, time_run, 1))  # Infectious stage 2
    Z3_PO = np.zeros((nsims, time_run, 1))  # Infectious stage 3
    Z4_PO = np.zeros((nsims, time_run, 1))

    Z1_PO[:, 0, 0] = initial_infected

    for i in range(nsims):
        beta_ = beta[i]
        gamma_ = gamma[i]
        N_ = N[i]

        Z1 = initial_infected
        Z2 = 0
        Z3 = 0
        Z4 = 0

        cum_time = 0
        while True:
            rates = np.array(
                [
                    beta_ * (N_ - Z1) * (Z1 - Z4) / (N_ - initial_infected),
                    3 * gamma_ * (Z1 - Z2),
                    3 * gamma_ * (Z2 - Z3),
                    3 * gamma_ * (Z3 - Z4),
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
            else:
                Z4 += 1

            if (Z1 - Z4) == 0:
                if remove_trivial:
                    if Z1 == initial_infected:
                        Z1_PO[i, :, 0] = 0
                        Z2_PO[i, :, 0] = 0
                        Z3_PO[i, :, 0] = 0
                        Z4_PO[i, :, 0] = 0

                        Z1_PO[i, 0, 0] = initial_infected

                        Z1 = initial_infected
                        Z2 = Z3 = Z4 = 0

                        cum_time = 0
                        continue
                Z1_PO[i, int(np.ceil(cum_time)) :, 0] = Z1
                Z2_PO[i, int(np.ceil(cum_time)) :, 0] = Z2
                Z3_PO[i, int(np.ceil(cum_time)) :, 0] = Z3
                Z4_PO[i, int(np.ceil(cum_time)) :, 0] = Z4
                break
    return Z1_PO, Z2_PO, Z3_PO, Z4_PO


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
def SIIIR_final_size(N, beta, gamma, initial_infected, max_len=100):
    Z1_PO, Z2_PO, Z3_PO, Z4_PO = SIIIR(
        N, beta, gamma, initial_infected, time_run=max_len
    )
    I = Z1_PO - Z4_PO
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
        Z4_PO[:, : (longest_obs + 1), :],
    )


# @numba.jit(nopython=True, fastmath=True)
def SI10R_final_size(N, beta, gamma, initial_infected, max_len=100):
    Z_POs = SI10R(N, beta, gamma, initial_infected, time_run=max_len)
    I = Z_POs[0] - Z_POs[-1]
    if np.all(I[:, -1, 0] == 0):
        longest_obs = 1
        for i in range(I.shape[0]):
            die_out = np.where(I[i, :, 0] == 0)[0][0]
            if die_out > longest_obs:
                longest_obs = die_out
    else:
        longest_obs = max_len
    return [Z[:, : (longest_obs + 1), :] for Z in Z_POs]


@numba.jit(nopython=True, fastmath=True)
def SEIR(N, beta, sigma, gamma, initial_infected, time_run, remove_trivial=False):
    nsims = beta.shape[0]
    Z1_PO = np.zeros((nsims, time_run, 1))
    Z2_PO = np.zeros((nsims, time_run, 1))
    Z3_PO = np.zeros((nsims, time_run, 1))

    Z1_PO[:, 0, 0] = initial_infected
    Z2_PO[:, 0, 0] = initial_infected

    for i in range(nsims):
        beta_ = beta[i]
        sigma_ = sigma[i]
        gamma_ = gamma[i]
        N_ = N[i]

        Z1 = initial_infected
        Z2 = initial_infected
        Z3 = 0

        cum_time = 0
        while True:
            rates = np.array(
                [
                    beta_ * (N_ - Z1) * (Z2 - Z3) / (N_ - initial_infected),
                    sigma_ * (Z1 - Z2),
                    gamma_ * (Z2 - Z3),
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
                        Z2_PO[i, 0, 0] = initial_infected

                        Z1 = Z2 = initial_infected
                        Z3 = 0

                        cum_time = 0
                        continue
                Z1_PO[i, int(np.ceil(cum_time)) :, 0] = Z1
                Z2_PO[i, int(np.ceil(cum_time)) :, 0] = Z2
                Z3_PO[i, int(np.ceil(cum_time)) :, 0] = Z3
                break
    return Z1_PO, Z2_PO, Z3_PO


@numba.jit(nopython=True, fastmath=True)
def SEIR_final_size(N, beta, sigma, gamma, initial_infected, max_len=100):
    Z1_PO, Z2_PO, Z3_PO = SEIR(
        N, beta, sigma, gamma, initial_infected, time_run=max_len
    )

    EpI = Z1_PO - Z3_PO
    if np.all(EpI[:, -1, 0] == 0):
        longest_obs = 1
        for i in range(EpI.shape[0]):
            die_out = np.where(EpI[i, :, 0] == 0)[0][0]
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
def LV(
    alpha,
    beta,
    gamma,
    delta,
    initial_predator,
    initial_prey,
    time_run,
    max_num_prey=10000,
):
    nsims = alpha.shape[0]
    X_PO = np.zeros((nsims, time_run, 1))
    Y_PO = np.zeros((nsims, time_run, 1))

    for i in range(nsims):
        alpha_ = alpha[i]
        beta_ = beta[i]
        gamma_ = gamma[i]
        delta_ = delta[i]
        X_PO[i, 0, 0] = initial_prey[i]
        Y_PO[i, 0, 0] = initial_predator[i]
        stop_flag = False

        X, Y = initial_prey[i], initial_predator[i]
        cum_time = 0
        while True:
            rates = np.array([alpha_ * X, beta_ * X * Y, delta_ * X * Y, gamma_ * Y])
            total_rate = np.sum(rates)
            t_new = np.random.exponential(1 / total_rate)
            # Number of integer time steps traversed
            time_steps_traversed = int(np.floor(cum_time + t_new) - np.floor(cum_time))
            if time_steps_traversed > 0:
                # Record partial observations if time step is traversed
                X_PO[
                    i,
                    int(np.floor(cum_time) + 1) : int(
                        np.floor(cum_time) + time_steps_traversed + 1
                    ),
                    0,
                ] = X
                Y_PO[
                    i,
                    int(np.floor(cum_time) + 1) : int(
                        np.floor(cum_time) + time_steps_traversed + 1
                    ),
                    0,
                ] = Y

                if stop_flag:
                    X_PO[i, int(np.ceil(cum_time) + 1) :, 0] = X
                    Y_PO[i, int(np.ceil(cum_time) + 1) :, 0] = 0.0
                    break

            cum_time = cum_time + t_new
            if cum_time > time_run:
                # Break the Gillespie algorithm
                break

            probs = rates / total_rate
            cdf = np.cumsum(probs)
            u = np.random.rand()

            if u < cdf[0]:
                X += 1
            elif u < cdf[1]:
                X -= 1
            elif u < cdf[2]:
                Y += 1
            else:
                Y -= 1

            if (X + Y) == 0:
                X_PO[i, int(np.ceil(cum_time)) :, 0] = X
                Y_PO[i, int(np.ceil(cum_time)) :, 0] = Y
                break

            if Y == 0 and X > max_num_prey:
                # Otherwise exponential growth occurs and simulations take forever
                stop_flag = True

    return X_PO, Y_PO


@numba.jit(nopython=True, fastmath=True)
def LV_immigration(
    alpha,
    beta,
    gamma,
    delta,
    prey_immigration,
    pred_immigration,
    initial_predator,
    initial_prey,
    predator_cc,
    prey_cc,
    time_run,
):
    nsims = alpha.shape[0]
    X_PO = np.zeros((nsims, time_run, 1))
    Y_PO = np.zeros((nsims, time_run, 1))

    for i in range(nsims):
        alpha_ = alpha[i]
        beta_ = beta[i]
        gamma_ = gamma[i]
        delta_ = delta[i]
        prey_immigration_ = prey_immigration[i]
        pred_immigration_ = pred_immigration[i]
        X_PO[i, 0, 0] = initial_prey[i]
        Y_PO[i, 0, 0] = initial_predator[i]

        X, Y = initial_prey[i], initial_predator[i]
        cum_time = 0
        while True:
            rates = np.array(
                [
                    (alpha_ * X + prey_immigration_) * (1 - X / prey_cc),
                    beta_ * X * Y,
                    (delta_ * X * Y + pred_immigration_) * (1 - Y / predator_cc),
                    gamma_ * Y,
                ]
            )
            total_rate = np.sum(rates)
            t_new = np.random.exponential(1 / total_rate)
            # Number of integer time steps traversed
            time_steps_traversed = int(np.floor(cum_time + t_new) - np.floor(cum_time))
            if time_steps_traversed > 0:
                # Record partial observations if time step is traversed
                X_PO[
                    i,
                    int(np.floor(cum_time) + 1) : int(
                        np.floor(cum_time) + time_steps_traversed + 1
                    ),
                    0,
                ] = X
                Y_PO[
                    i,
                    int(np.floor(cum_time) + 1) : int(
                        np.floor(cum_time) + time_steps_traversed + 1
                    ),
                    0,
                ] = Y

            cum_time = cum_time + t_new
            if cum_time > time_run:
                # Break the Gillespie algorithm
                break

            probs = rates / total_rate
            cdf = np.cumsum(probs)
            u = np.random.rand()

            if u < cdf[0]:
                X += 1
            elif u < cdf[1]:
                X -= 1
            elif u < cdf[2]:
                Y += 1
            else:
                Y -= 1

    return X_PO, Y_PO


@numba.jit(nopython=True, fastmath=True)
def MG1(theta1, theta2, theta3, time_run):
    nsims = theta1.shape[0]
    X_PO = np.zeros((nsims, time_run, 1))

    for i in range(nsims):
        theta1_ = theta1[i]
        theta2_ = theta2[i]
        theta3_ = theta3[i]
        X = 0

        passed_time = 0
        while True:
            if X == 0:
                arrival_time = np.random.exponential(1 / theta3_)
                time_steps_traversed = int(
                    np.floor(passed_time + arrival_time) - np.floor(passed_time)
                )
                if time_steps_traversed > 0:
                    # Record partial observations if time step is traversed
                    X_PO[
                        i,
                        int(np.floor(passed_time) + 1) : int(
                            np.floor(passed_time) + time_steps_traversed + 1
                        ),
                        0,
                    ] = X
                passed_time += arrival_time
                if passed_time > time_run:
                    break
                X += 1
            else:
                service_time = theta1_ + (theta2_ - theta1_) * np.random.rand()
                time_left = service_time
                while True:
                    arrival_time = np.random.exponential(1 / theta3_)
                    if arrival_time < time_left:
                        time_steps_traversed = int(
                            np.floor(passed_time + arrival_time) - np.floor(passed_time)
                        )
                        if time_steps_traversed > 0:
                            # Record partial observations if time step is traversed
                            X_PO[
                                i,
                                int(np.floor(passed_time) + 1) : int(
                                    np.floor(passed_time) + time_steps_traversed + 1
                                ),
                                0,
                            ] = X
                        passed_time += arrival_time
                        if passed_time > time_run:
                            break
                        time_left -= arrival_time
                        X += 1
                    else:
                        break
                time_steps_traversed = int(
                    np.floor(passed_time + time_left) - np.floor(passed_time)
                )
                if time_steps_traversed > 0:
                    # Record partial observations if time step is traversed
                    X_PO[
                        i,
                        int(np.floor(passed_time) + 1) : int(
                            np.floor(passed_time) + time_steps_traversed + 1
                        ),
                        0,
                    ] = X
                X -= 1
                passed_time += time_left
                if passed_time > time_run:
                    break
    return X_PO
