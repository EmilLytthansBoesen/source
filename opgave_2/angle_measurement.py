import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ipywidgets import widgets, interactive
from IPython.display import display

from opgave_2.shapes import Shape, Ball
from opgave_2.falling import Data, GRAVITATIONAL_ACCELERATION, calculate_fall_curve, calculate_air_resistance
from opgave_2.user_interface import AngleSlider, MaterialDropDown

DEFAULT_HEIGHT: float   = 1.0  # meter
DEFAULT_SPEED: float    = 10.0  # meter/seconds
DEFAULT_RADIUS: float   = 0.10  # meter


RADIANS_PR_DEGREE: float = np.pi / 180
KILOGRAM_M_CUBES_OR_GCM: float = 1_000


class AngleMeasurementApp:
    height: float
    throwing_speed: float
    shape: Shape

    def __init__(self, height: float=DEFAULT_HEIGHT, throwning_speed: float=DEFAULT_SPEED, shape: Shape=None):
        self.height         = height
        self.throwing_speed = throwning_speed
        self.shape          = shape if shape is not None else Ball(radius=DEFAULT_RADIUS)


    def generate_plot(self, angle_degree: float, density_gcm: float) -> None:
        """Plot the fall curve for a given angle and material density"""
        angle = RADIANS_PR_DEGREE * angle_degree
        density = KILOGRAM_M_CUBES_OR_GCM * density_gcm
        
        self.prepare_plot()
        self.plot_maximum_distance_curve()
        distance = self.plot_air_resistance_curve(angle, density)
        add_title(distance)


    def prepare_plot(self) -> None:
        axis_limit_x = self.maximum_distance_without_air_resistance
        axis_limit_y = self.maximum_height_without_air_resistance

        define_limits(axis_limit_x, axis_limit_y)
        add_coordiante_system()
        add_labels()


    def plot_air_resistance_curve(self, angle: float, density: float) -> None:
        """Plot the fall curve assuming air resistance, density of the object, and a given angle"""
        _, _, falldata = calculate_fall_curve(angle, 
                                              air_resistance=calculate_air_resistance(self.shape, density), 
                                              throwing_speed=self.throwing_speed, 
                                              initial_height=self.height)
        
        plt.plot(falldata.x, falldata.y, label="Beregnerede kastekurve")
        return falldata.x.max()


    def plot_maximum_distance_curve(self) -> None:
        """Plot the throw assuming no air resistance and throwing at the optimal angle of 45 degrees"""
        x = np.linspace(0, self.maximum_distance_without_air_resistance)
        plt.plot(x, self.maximum_distance_curve(x), '--g', label='Teoretisk maximalkurve')


    def maximum_distance_curve(self, x: np.ndarray[float]) -> np.ndarray[float]:
        return self.height + x - GRAVITATIONAL_ACCELERATION / self.throwing_speed**2 * x**2
    

    @property
    def maximum_distance_without_air_resistance(self) -> float:
        alpha = - GRAVITATIONAL_ACCELERATION / self.throwing_speed**2
        beta  = 1
        gamma = self.height
        delta = beta**beta - 4*alpha*gamma
        return (-beta - np.sqrt(delta)) / (2 * alpha)


    @property
    def maximum_height_without_air_resistance(self) -> float:
        return self.throwing_speed**2 / (2 * GRAVITATIONAL_ACCELERATION) + self.height



def run_angle_experiment() -> None:
    app = AngleMeasurementApp()
    widgets.interact(app.generate_plot, angle_degree=AngleSlider, density_gcm=MaterialDropDown)


def define_limits(axis_limit_x: float, axis_limit_y: float) -> None:
    plt.xlim((-0.25, axis_limit_x))
    plt.ylim((-0.25, axis_limit_y))


def add_labels(fontsize: float=18) -> None:
    plt.xlabel("Afstand [m]", fontsize=fontsize)
    plt.ylabel("Højde [m]", fontsize=fontsize)



def add_coordiante_system() -> None:
    plt.axline((0, 0), (1, 0), color='k')  # x-axis
    plt.axline((0, 0), (0, 1), color='k')  # y-axis
    plt.grid()


def add_title(distance: float, *, fontsize=18) -> None:
    plt.title(f"Kastelængde: {distance:0.2f} [m]", fontsize=fontsize)