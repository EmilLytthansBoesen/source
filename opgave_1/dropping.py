import numpy as np


_LOG_2: float = np.log(2.0)
GRAVITATIONAL_ACCELERATION: float = 9.82 # m /s^2
AIR_DENSITY = 1.293                      # kg / m^3


def falltime_with_air_resistance(mass: np.ndarray[float], drag_coefficient: float, height: float, cross_section: float) -> np.ndarray[float]:
    air_resistance = 0.5 * drag_coefficient * AIR_DENSITY * cross_section

    terminal_speed  = np.sqrt(mass * GRAVITATIONAL_ACCELERATION / air_resistance)
    time_scale      = terminal_speed / GRAVITATIONAL_ACCELERATION
    length_scale    = terminal_speed * time_scale

    return time_scale * falltime_scalar_approx(height / length_scale)


def falltime_gallileo(mass: np.ndarray[float], height: float) -> np.ndarray[float]:
    falltime = np.sqrt(2.0 * height / GRAVITATIONAL_ACCELERATION)
    return falltime * np.ones(mass.shape, dtype=float)



def falltime_scalar_func(x: np.ndarray[float]) -> np.ndarray[float]:
    """Calculate the unit-less fall time from a unit-less height

    Args:
        x (np.ndarray[float]): Unit-less height. Height of the fall scaled by the charecteristic height

    Returns:
        t (np.ndarray[float]): Unit-less fall-time. Fall-time scaled by the charecteristic time
    """
    return np.arccosh(np.exp(x))


def falltime_scalar_approx(x: np.ndarray[float], *, large_x_transition: float=10) -> np.ndarray[float]:
    """Approximation of the unit-less fall-time curve using the large-x approxmation to avoid overflow errors

    Args:
        x (np.ndarray[float]): Unit-less height. Height of the fall scaled by the charecteristic height
        large_x_transition (float, optional): x-value for which the approxmation is used. Defaults to 10.
    """
    small_x = np.where(x < large_x_transition, x, 0)
    large_x = np.where(x >= large_x_transition, x, 0)
    
    falltime_small_x = (small_x != 0) * falltime_scalar_func(small_x)
    falltime_large_x = (large_x != 0) * falltime_scalar_large_x_approx(large_x)
    
    return falltime_small_x + falltime_large_x

    return np.where(x < large_x_transition, falltime_scalar_func(x), falltime_scalar_large_x_approx(x))


def falltime_scalar_large_x_approx(x: np.ndarray[float]) -> np.ndarray[float]:
    """Large x approxmation of the fall time calculation"""
    return _LOG_2 + x


def _test() -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3)
    x = np.linspace(0, 10)

    ax[0].plot(x, np.arccosh(x), label='arccosh')
    ax[0].plot(x, np.log(2.0 * x), label='log')
    ax[0].legend()

    ax[1].plot(x, falltime_scalar_func(x), label='$f^{-1}$')
    ax[1].plot(x, falltime_scalar_approx(x), label='$g$')
    ax[1].legend()

    ax[2].plot(x, falltime_scalar_approx(x) - falltime_scalar_func(x), label='Error')

    plt.show()


if __name__ == '__main__':
    _test()