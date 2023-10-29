import opgave_2.shapes as shapes
import numpy as np
import matplotlib.pyplot as plt

from opgave_2.shapes import Shape
from opgave_2.falling import calculate_fall_curve
from matplotlib.axes import Axes


THROW_ANGLE: float = 45
AIR_DENSITY = 1.293  # kg / m^3


def price_pr_meter(shape: Shape, density: float, angle: float=THROW_ANGLE) -> tuple[float, float]:
    """Calculate the price coefficient of a given shape 

    Args:
        shape (Shape):      Shape of projectile
        density (float):    Density of the material
        angle (float, optional): Angle of the projectile when fired. Defaults to 45 degrees.

    Returns:
        Distance (float):           Distance flowns by the object
        Price coefficient (float):  Volume pr. distance moved
    """
    mass = shape.mass(density)
    air_resistance = 0.5 * shape.drag_coefficient * shape.cross_area * AIR_DENSITY / mass

    _, dist, __ = calculate_fall_curve(angle, air_resistance)
    
    price_coeff = mass / dist
    return dist, price_coeff



def price_results(radii: np.ndarray[float], distance: np.ndarray[float], price_coeff: np.ndarray[float]) -> None:
    fig, left_ax = plt.subplots()
    right_ax = left_ax.twinx()

    LEFT_COLOR = "tab:red"
    RIGHT_COLOR = "tab:blue"

    left_ax.set_xlabel(f"Radius [m]")

    left_ax.plot(radii, distance, '--*', color=LEFT_COLOR)
    left_ax.set_ylabel("Distance moved [m]", color=LEFT_COLOR)
    left_ax.tick_params(axis='y', labelcolor=LEFT_COLOR)

    right_ax.plot(radii, price_coeff, '--*', color=RIGHT_COLOR)
    right_ax.set_ylabel("Price coefficient [m$^2$]")
    right_ax.tick_params(axis='y', labelcolor=RIGHT_COLOR)

    plt.show()
    



def main() -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    density = 1000
    radii = np.linspace(1e-3, 0.1)
    
    
    def price_of_shape(radius: float) -> float:
        return price_pr_meter(shapes.Ball(radius), density)
        

    dist_array = []
    price_coeff_array = []
    for radius in radii:
        dist, price_coeff = price_of_shape(radius)
        dist_array.append(dist)
        price_coeff_array.append(price_coeff)
    
    price_results(radii, dist_array, price_coeff_array)

    


if __name__ == '__main__':
    main()