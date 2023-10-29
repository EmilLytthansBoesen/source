"""
This codebase is used the the internship week of DTU Energy 2023
================================================================

"""
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from typing import Callable

from opgave_2.shapes import Shape

INITIAL_HEIGHT = 1      # Measured in meters
THROWING_SPEED = 10     # Measured in meters pr. second
ANGLE_RANGE = [0, 90]   # Measured in degrees

GRAVITATIONAL_ACCELERATION = 9.82   # m / s^2
AIR_DENSITY = 1.293     # Measured in kg/m^3
TIME_STEP   = 0.001     # Measured in seconds


class Data:
    x: np.ndarray[float]    = np.empty((0, ), dtype=float)
    y: np.ndarray[float]    = np.empty((0, ), dtype=float)
    vx: np.ndarray[float]   = np.empty((0, ), dtype=float)
    vy: np.ndarray[float]   = np.empty((0, ), dtype=float)

    def __init__(self, x: float, y: float, vx: float, vy: float):
        self.add(x, y, vx, vy)

    
    def add(self, x: float, y: float, vx: float, vy: float) -> None:
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)
        self.vx = np.append(self.vx, vx)
        self.vy = np.append(self.vy, vy)

    
    def distance(self) -> float:
        return self.x[-1]
        


def calculate_fall_curve(angle: float, air_resistance: float, throwing_speed: float=None, initial_height: float=None, time_step:float=None) -> tuple[float, float, Data]:
    """Calculate the falltime and the fall curve of a thrown object

    Args:
        angle (float):  Angle of incline of the thrown object. 0 <= angle <= 90
        air_resistance (float): Coefficient of air resistance being "b/m" and measured in 1/m

    Returns:
        air_time (float):           The time the object is in the air
        thrown_distance (float):    The distance the object flies along the x-direction
        np.ndarray[float, float]:   (x, y, vx, vy) coordiantes of the thrown object 
    """
    throwing_speed  = THROWING_SPEED if throwing_speed is None else throwing_speed
    initial_height  = INITIAL_HEIGHT if initial_height is None else initial_height
    time_step       = TIME_STEP if time_step is None else time_step
        

    x = 0
    y = initial_height
    vx = throwing_speed * np.cos(angle)
    vy = throwing_speed * np.sin(angle)
    data = Data(x, y, vx, vy)
    
    time = 0
    euler_step_partial = partial(euler_step, dt=time_step, coefficient_of_resistance=air_resistance)

    while y > 0:
        dx1, dy1, dvx1, dvy1 = euler_step_partial(x, y, vx, vy)
        dx2, dy2, dvx2, dvy2 = euler_step_partial(x + 0.5*dx1, y + 0.5*dy1, vx + 0.5*dvx1, vy + 0.5*dvy1)
        dx3, dy3, dvx3, dvy3 = euler_step_partial(x + 0.5*dx2, y + 0.5*dy2, vx + 0.5*dvx2, vy + 0.5*dvy2)
        dx4, dy4, dvx4, dvy4 = euler_step_partial(x + dx3, y + dy3, vx + dvx3, vy + dvy3)
        
        x  += rk_step(dx1, dx2, dx3, dx4)
        y  += rk_step(dy1, dy2, dy3, dy4)
        vx += rk_step(dvx1, dvx2, dvx3, dvx4)
        vy += rk_step(dvy1, dvy2, dvy3, dvy4)

        time += TIME_STEP
        data.add(x, y, vx, vy)

    return time, data.distance(), data



def rk_step(k1: float, k2: float, k3: float, k4: float) -> float:
    return (k1 + 2*k2 + 2*k3 + k4) / 6


def euler_step(x: float, y: float, vx: float, vy: float, dt: float, coefficient_of_resistance: float) -> tuple[float, float, float, float]:
    """Step forward using the euler-algorithm of solving differential equations

    Args:
        x (float): Distance moved along the x-direction
        y (float): Distance moved along the y-direction
        vx (float): Speed along the x-direction
        vy (float): Speed along the y-direction
        dt (float): Step size of the simulation
        coefficient_of_resistance (float): Step size of the simulation

    Returns:
        dx (float): Change in distance along x-direction
        dy (float): Change in distance along x-direction
        dvx (float): Change in speed along the x-direction
        dvy (float): Change in speed along the y-direction
    """
    speed = np.sqrt(vx*vx + vy*vy)
    vx_unit = vx / speed
    vy_unit = vy / speed
    resistance_acceleration = coefficient_of_resistance * speed * speed

    dx  = TIME_STEP * vx
    dy  = TIME_STEP * vy
    dvx = TIME_STEP * (-vx_unit * resistance_acceleration)
    dvy = TIME_STEP * (-GRAVITATIONAL_ACCELERATION - vy_unit * resistance_acceleration)
    
    return dx, dy, dvx, dvy
    

def calculate_air_resistance(shape: Shape, density: float) -> float:
    air_ceofficient = 0.5 * shape.drag_coefficient * AIR_DENSITY * shape.cross_area
    mass = shape.mass(density)
    return air_ceofficient / mass



def invert_monotonic_increasing_function(f: Callable[[float], float], lower_bound: float|Callable[[float], float], upper_bound: float|Callable[[float], float]) -> Callable[[np.ndarray[float]], np.ndarray[float]]:
    """Assuming that the provided function is monotonic increasing, then the inverse evaluated at "x" can be calculated"""
    if isinstance(lower_bound, (int, float)):
        lower_bound_value = lower_bound
        lower_bound = lambda _: lower_bound_value
    if isinstance(upper_bound, (int, float)):
        upper_bound_value = upper_bound
        upper_bound = lambda _: upper_bound_value

    def f_inverse(y: float) -> float:
        return inverse_binary_search(y, f, lower_bound(y), upper_bound(y))

    return np.vectorize(f_inverse)
    
    
def inverse_binary_search(y: np.ndarray[float], f: Callable[[float], float], lower_bound: float, upper_bound: float, steps: int=10) -> float:
    """Find a value x such that y = f(x) for a monotonic increasing function "f".

    Args:
        y (np.ndarray[float]): Value for which the inverse of f is to be called on
        f (callable[[float], float]): Function that is to be inverted      
        lower_bound (float): Lower limit of the binary search
        upper_bound (float): Upper limit on the binary search
        steps (int, optional): Number of steps in the search. Defaults to 10.

    Returns:
        x : (float) Value that satisfies x = f(y)
    """
    x_left  = lower_bound
    x_right = upper_bound

    for _ in range(steps):
        x_mid = 0.5 *(x_left + x_right)
        y_mid = f(x_mid)

        if y_mid == y:
            return x_mid
        if (y_mid < y):
            x_left = x_mid
        else:
            x_left = x_mid

    return x_mid
    

def _test() -> None:
    radius = 0.1
    aluminum_density = 2700 # kg / m^3
    drag_coefficient = 0.47

    area = np.pi * radius**2
    mass = aluminum_density * 4/3 * np.pi * radius**3

    b_coeff = 0.5 * drag_coefficient * AIR_DENSITY * area

    resistance_coefficient = b_coeff / mass

    time, dist, data = calculate_fall_curve(45, resistance_coefficient)
    plt.plot(data.x, data.y)    


def _test2() -> None:
    def nonlinear_function(x: np.ndarray[float]) -> np.ndarray[float]:
        return x + np.log(0.5 * (1 + np.exp(-2*x)))

    x = np.linspace(0, 10)
    f_invese = invert_monotonic_increasing_function(nonlinear_function, 0, lambda x: x)

    fig, ax = plt.subplots(2, 1)

    ax[0].plot(x, nonlinear_function(x), label=r"$f$")
    ax[0].plot(x, f_invese(x), label=r"$f^{-1}$")
    ax[0].legend()

    ax[1].plot(x, f_invese(nonlinear_function(x)) - x, label=r"$f^{-1} \circ f(x)$")
    ax[1].plot(x, nonlinear_function(f_invese(x)) - x, label=r"$f \circ f^{-1}(x)$")
    ax[1].legend()

if __name__ == '__main__':
    _test()
    _test2()
    plt.show()