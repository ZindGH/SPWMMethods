import numpy as np
import pandas as pd

from matplotlib import rcParams
import matplotlib
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import scipy
from scipy import signal
import unittest

"""Testing"""


class TestSomeSine(unittest.TestCase):
    def testArray(self, array, F_tri: int, pwm_T: int, T_number: float):
        subList = [array[n:n + 2] for n in range(0, len(array), 2)]
        self.assertFalse(any([abs(pair[1] - pair[0]) > pwm_T for pair in subList]), f"NE RABOTAET! \n"
                                                                                    f"PWM VIOLATED PERIOD BORDERS!")
        self.assertTrue((len(array) % (F_tri / 400)) == 0, f"NE RABOTAET! \n"
                                                           f"POHODU METHOD POSCHITAL NE TO KOLVO ZNACHENIY: {len(array)}")


"""Testing"""


def makeSineReworked(amp: float = 0.90, pwm_T: int = 10000, F_tri: int = 2800, sin_shift: int = 0,
                     number_of_T: float = 0.5, method: str = "PCPS", tri_sign=True, pwm_inv=0):
    F_sin = 400
    Tsin = (1 / F_sin)
    match method:
        case "PCPS":
            n_samples = int(number_of_T * pwm_T * F_tri / 400)
        case "PNCDS":
            n_samples = int(number_of_T * pwm_T * F_tri / 400)

    t = np.linspace(0, Tsin * number_of_T, n_samples)
    Td = t[1] - t[0]

    match method:
        case ("PCPS"):
            sinus = amp * np.sin(2 * np.pi * F_sin * (t - sin_shift / 360))
            if tri_sign:
                triangle = 1 / 2 * (signal.sawtooth(2 * np.pi * F_tri * t, 0.5) + 1)
                sinus = np.abs(sinus)
            else:
                triangle = signal.sawtooth(2 * np.pi * F_tri * t, 0.5)

            plt.plot(t / Td, triangle)  #

            plt.plot(t / Td, sinus)

        case ("PNCDS"):
            triangle = 1 * (signal.sawtooth(2 * np.pi * F_tri * t, 0.5))
            plt.plot(t / Td, triangle)

            sinus = amp * np.sin(2 * np.pi * F_sin * (t - sin_shift / 360))
            sinusN = -amp * np.sin(2 * np.pi * F_sin * (t - sin_shift / 360))

            plt.plot(t / Td, sinus)
            plt.plot(t / Td, sinusN)

        case _:
            raise f"tvoi method: {method} --- NOT IMPLEMENTED"

    res_t = list()
    sign_PCPS: int = 1
    sign_PNCDS_N = 1  #
    sign_PNCDS = 1
    # First check:
    match method:
        case "PCPS":
            if ((sinus[1] - triangle[1]) * sign_PCPS) > 0:
                sign_PCPS *= -1
        case "PNCDS":
            if ((sinus[1] - triangle[1]) * sign_PNCDS) > 0:
                sign_PNCDS = -1
            if ((sinusN[1] - triangle[1]) * sign_PNCDS_N) > 0:
                sign_PNCDS_N = -1

    for i in range(2, len(t) - 1):
        match method:
            case "PCPS":
                if ((sinus[i] - triangle[i]) * sign_PCPS) > 0:
                    res_t.append(i)
                    sign_PCPS *= -1
            case "PNCDS":
                if ((sinus[i] - triangle[i]) * sign_PNCDS) > 0:
                    res_t.append(i)
                    sign_PNCDS *= -1
                if ((sinusN[i] - triangle[i]) * sign_PNCDS_N) > 0:
                    res_t.append(i)
                    sign_PNCDS_N *= -1

    print(f"array: {res_t}\narray_len: {len(res_t)}")

    pwm_temp_val = [i for i in res_t][0:]
    pair_centers = list()
    pair_diffs = list()
    new_pwm = list()
    if pwm_inv:
        subList = [pwm_temp_val[n:n + 2] for n in range(0, len(pwm_temp_val), 2)]
        for pair in subList:
            center = round((pair[0] + pair[1]) / 2)
            diff = pwm_T - abs(pair[1] - pair[0])

            new_pwm.append(round((center - diff / 2)))
            new_pwm.append(round((center + diff / 2)))
            pwm_temp_val = new_pwm

    res_pwm = list()
    flag_on = False

    for i in range(n_samples):
        if i in pwm_temp_val:
            flag_on = not flag_on
        match method:
            case "PCPS":
                pwm_amp = 0.5
                if flag_on:
                    res_pwm.append(pwm_amp)
                else:
                    if not tri_sign:
                        res_pwm.append(-pwm_amp)
                        continue
                    res_pwm.append(0)
            case "PNCDS":
                pwm_amp = 0.3
                if flag_on:
                    res_pwm.append(pwm_amp)
                else:
                    res_pwm.append(-pwm_amp)

    res_pwm = np.array(res_pwm)

    plt.plot(t / Td, res_pwm)

    return res_t, Td


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
    tester = TestSomeSine()

    pwm_T = 10000
    F_tri = 2800
    number_of_T = 2
    # time_array, Td = makeSineReworked(amp=0.89, pwm_T=10000, F_tri=5600, sin_shift=240,
    #                                   number_of_T=1, method="PCPS", tri_sign=0, pwm_inv=0)

    time_array, Td = makeSineReworked(amp=0.50, pwm_T=pwm_T, F_tri=F_tri, sin_shift=240,
                                      number_of_T=number_of_T, method="PNCDS", pwm_inv=0)
    # time_array.append(1)
    # time_array.append(100)
    tester.testArray(time_array, F_tri, pwm_T, number_of_T)
    time_array = [spwm_t % pwm_T for spwm_t in time_array]
    ds = pwm_T - time_array[-1] + time_array[0]

    print(f"DeadShift: {ds}\nDeadShift_ms: {ds * Td * 1000}")

    print_c_array(time_array, wideness=4)
    plt.show()
