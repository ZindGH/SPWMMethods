import numpy as np
import pandas as pd

from matplotlib import rcParams
import matplotlib
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import scipy
from scipy import signal

def makeSine(n_samples: int = 10000, shift_tri: int = 25, F_tri: int = 2800, eps: float = 0.02, sin_shift: int = 100,
             method: str = "PosCarrier PosSine SPWM"):
    amp = 3
    F_sin = 400
    # F_tri = 3000
    tan = 15 * np.power(10, 3)

    Tsin_half = 1 / 2 * (1 / F_sin)

    t = np.linspace(0, Tsin_half, n_samples)
    Td = t[1] - t[0]
    mflag = None  # Method abbr
    match method:
        case ("PosCarrier PosSine SPWM"):
            triangle = 4 / 2 * (signal.sawtooth(2 * np.pi * F_tri * (t - sin_shift * Td), 0.5) + 1)
            triangle = np.concatenate(([0] * shift_tri, triangle))
            triangle = np.roll(triangle, shift_tri)[shift_tri:]

            plt.plot(t / Td, triangle)  #

            sinus = amp * np.sin(2 * np.pi * F_sin * t)

            plt.plot(t / Td, sinus)

            mflag = "PCPS"
        case ("PNCarrier DualSim SPWM"):

            triangle = 6 / 2 * (signal.sawtooth(2 * np.pi * F_tri * (t - sin_shift * Td), 0.5))
            triangle = np.concatenate(([0] * shift_tri, triangle))
            triangle = np.roll(triangle, shift_tri)[shift_tri:]

            plt.plot(t / Td, triangle)  #

            sinus = amp * np.sin(2 * np.pi * F_sin * t)
            sinusN = -amp * np.sin(2 * np.pi * F_sin * t)

            plt.plot(t / Td, sinus)
            plt.plot(t / Td, sinusN)
            mflag = "PNCDS"

        case _:
            raise f"tvoi method: {method} --- NOT IMPLEMENTED"

    # match phase_sin:
    #     case '0':
    #         t_sin = t
    #         t_tri = np.pad(t, (shift_tri, 0), mode='constant')
    #     case '120':
    #         t_sin = np.pad(t, ((1 / F_sin) / 3 - Td, 0), mode='constant')
    #         t_tri = np.pad(t, ((1 / F_sin) / 3 - Td + shift_tri, 0), mode='constant')
    #     case '240':
    #         t_sin = np.pad(t, (2 * (1 / F_sin) / 3 - Td, 0), mode='constant')
    #         t_tri = np.pad(t, (2 * (1 / F_sin) / 3 - Td + shift_tri, 0), mode='constant')

    res_t = list()
    just_got_value = 0
    for i in range(len(t)):
        if i <= shift_tri:
            continue
        if 0 < just_got_value < n_samples / (1000 * 1.5):
            just_got_value += 1
            continue
        elif just_got_value >= n_samples / (1000 * 1.5) or just_got_value == 0:
            just_got_value = 0
        if mflag == "PNCDS":
            if (abs(triangle[i] - sinus[i])) <= eps or (abs(triangle[i] - sinusN[i])) <= eps:
                res_t.append((triangle[i], i))
                just_got_value += 1
        elif mflag == "PCPS":
            if (abs(triangle[i] - sinus[i])) <= eps:
                res_t.append((triangle[i], i))
                just_got_value += 1

    print(f"array: {res_t}\narray_len: {len(res_t)}")
    # print(min_debug_val)
    pwm_temp_val = [i for _, i in res_t][0:]
    res_pwm = list()
    flag_on = False
    for i in range(n_samples):
        if i in pwm_temp_val:
            flag_on = not flag_on
        if flag_on:
            res_pwm.append(1)
        else:
            res_pwm.append(0)

    res_pwm = np.array(res_pwm)
    plt.plot(t / Td, res_pwm)

    return [round(x, 4) for x, _ in res_t], [i for _, i in res_t], Td


def print_c_array(array, wideness: int = 8):
    len_array = len(array)
    print("{")
    for i, el in enumerate(array):
        if i == len_array - 1:
            print(f"{el}", end='')
            continue
        if (i + 1) % wideness == 0:
            print(f"{el}, \n", end='')
            continue
        print(f"{el}, ", end='')
    print("\n}")






if __name__ == '__main__':
    T_samples = 10000
    _, time_array, Td = makeSine(n_samples=14 * T_samples, F_tri=2800 * 2, shift_tri=0, eps=0.001,
                                 sin_shift=T_samples)

    # _, time_array, Td = makeSine(n_samples=T_samples * 7, F_tri=2800, shift_tri=0, eps=0.003, sin_shift=0,
    #                              method="PNCarrier DualSim SPWM")

    time_array = [spwm_t % T_samples for spwm_t in time_array]
    ds = T_samples - time_array[-1] + time_array[0]

    print(f"DeadShift: {ds}\nDeadShift_ms: {ds * Td * 1000}")

    print_c_array(time_array, wideness=4)
    plt.show()

