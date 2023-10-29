import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from dataclasses import dataclass
from functools import partial
from opgave_1.dropping import falltime_gallileo, falltime_with_air_resistance

from scipy.optimize import curve_fit

METERS_PR_CENTIMITER: float = 1 / 100
KILOGRAMS_PR_GRAM: float    = 1 / 1000
GRAVITATIONAL_ACCELERATION: float = 9.82 # m /s^2
AIR_DENSITY = 1.293  # kg / m^3

_LOG_2 = np.log(2.0)

@dataclass
class CupCakeExperiment:
    height_si: float
    base_mass_si: float
    diameter_si: float

    def __init__(self, height_si: float, base_mass: float, diameter: float) -> None:
        self.height_si = height_si
        self.base_mass_si = base_mass
        self.diameter_si = diameter
        
    
    @property
    def cross_section(self) -> float:
        return np.pi * (0.5 * self.diameter_si)**2


def get_experimental_setup(height_in_cm: float, base_mass_in_g: float, diameter_in_cm: float, *, verbose: bool=True) -> CupCakeExperiment:
    if height_in_cm is Ellipsis:
        height_in_cm    = float(input("Hvor højt lod i kageformerne falde fra? - Indtast venligst i cm"))
    if base_mass_in_g is Ellipsis:
        base_mass_in_g  = float(input("Hvor meget vejede kageformerne? - Indtast venligst i g"))
    if diameter_in_cm is Ellipsis:
        diameter_in_cm  = float(input("Hvad er den største diameter for kageformen - Indtest venligst i cm"))

    if verbose:
        print("Okay!")
        print("=========================================================================")
        print(f"I lod dem falde fra {height_in_cm} cm's højde.")
        print(f"En enkelt kageform vejede {base_mass_in_g} g.")
        print(f"Diameteren af kageformen er {diameter_in_cm} cm.")
        print("Hvis dette ikke er rigtigt, kør cellen igen og skriv jeres nye resultater.")
        print("=========================================================================")

    height      = METERS_PR_CENTIMITER * height_in_cm
    diameter    = METERS_PR_CENTIMITER * diameter_in_cm
    base_mass   = KILOGRAMS_PR_GRAM * base_mass_in_g

    return CupCakeExperiment(height, base_mass, diameter)



def draw_results(number_of_forms: list[int], falltime: list[float], experimental_setup: CupCakeExperiment, *, ax: Axes=None, fit_drag_coeff: float=None, loglog: bool=False) -> None:
    """Show the data collected by the student assuming that the shape of the cakeform is a truncated cone
    
    Args:
        number_of_forms (list[int]): Number of forms dropped by the user
        falltime (list[float]): Falltime measured by the user
        experimental_setup (CupCakeExperiment): Contains information about the fall experiment

    Kwargs:
        ax (Axes, optional): If not provided, a new plot is generated
        fit_drag_coeff (None | float, optional): If float, the fit is calculated using the given value as theoretical drag coefficient.
        loglog (bool, optional): If true the plot has logarithmic axis. Defaults to False.
    """
    if ax is None:
        _, ax = plt.subplots()

    mass     = experimental_setup.base_mass_si * np.asarray(number_of_forms)
    falltime = np.asarray(falltime)
    
    # Plot theoretical results
    plot_gallileo_curve(ax, mass, experimental_setup)
    plot_theoretical_curve(ax, mass, experimental_setup, with_average=(fit_drag_coeff is None))
    plot_data(ax, mass, falltime, experimental_setup, fit_drag_coeff=fit_drag_coeff)
    
    ax.grid()
    ax.set_title("Hvordan afhænger faldtid af masse?")
    ax.set_xlabel("Masse af genstanden i kg.")
    ax.set_ylabel("Faldtiden målt i sekunder.")

    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        ax.set_xscale("linear")
        ax.set_yscale("linear")

    ax.legend()



def plot_gallileo_curve(ax: Axes, mass: np.ndarray[float], experimental_setup: CupCakeExperiment) -> None:
    """Plot the falltime's mass dependence according to Gallileo's fall experiment"""
    falltime = falltime_gallileo(mass, experimental_setup.height_si)
    ax.plot(mass, falltime, '-r', label="Galileo's faldlov")



def plot_theoretical_curve(ax: Axes, mass: np.ndarray[float], experimental_setup: CupCakeExperiment, *, with_average: bool=True) -> None:
    """Plot the falltime's mass dependecy using air resistance"""
    DRAG_COEFFICIENT_CONE = 0.50
    DRAG_COEFFICIENT_CUBE = 1.05
    
    continuous_mass = np.linspace(min(mass), max(mass))
    falltime_minimum = falltime_with_air_resistance(continuous_mass, DRAG_COEFFICIENT_CONE, experimental_setup.height_si, experimental_setup.cross_section)
    falltime_maximum = falltime_with_air_resistance(continuous_mass, DRAG_COEFFICIENT_CUBE, experimental_setup.height_si, experimental_setup.cross_section)
    
    ax.fill_between(continuous_mass, falltime_minimum, falltime_maximum)
    ax.plot(continuous_mass, falltime_minimum, '--b')
    ax.plot(continuous_mass, falltime_maximum, '--b')
    
    if with_average:
        falltime_average = 0.5 * (falltime_maximum + falltime_minimum)
        ax.plot(continuous_mass, falltime_average, '-b', label="Teoretisk faldtid")



def plot_data(ax: Axes, mass: list[float], measured_falltimes: list[float], experimental_setup: CupCakeExperiment, *, fit_drag_coeff: float=None) -> None:
    """Plot the measured data using points. If drag_coefficient is provided the fit is also drawn

    Args:
        ax (Axes): _description_
        mass (list[float]): _description_
        measured_falltimes (list[float]): _description_
        experimental_setup (CupCakeExperiment): _description_
        fit_drag_coeff (float, optional): _description_. Defaults to None.
    """
    if fit_drag_coeff is not None:
        continuous_mass = np.linspace(min(mass), max(mass))
        falltime_fit = falltime_with_air_resistance(continuous_mass, fit_drag_coeff, experimental_setup.height_si, experimental_setup.cross_section)
        ax.plot(continuous_mass, falltime_fit, '-b', label="Teoretisk faldtid")

    # Guardclosing does not work here because the data points needs to be plotted after the teoretical values
    ax.plot(mass, measured_falltimes, '*', color='purple', label="Data")


def calculate_drag_coefficient_from_fit(number_of_forms: list[int], measured_falltime: list[float], experiment_setup: CupCakeExperiment) -> float:
    mass = experiment_setup.base_mass_si * np.asarray(number_of_forms)

    falltime_curves = partial(falltime_with_air_resistance, height=experiment_setup.height_si, cross_section=experiment_setup.cross_section)
    drag_coefficient, covariance = curve_fit(falltime_curves, mass, measured_falltime)

    return drag_coefficient[0], np.sqrt(covariance[0,0])


def _secret_results(drag_coefficient: float, experiment: CupCakeExperiment) -> tuple[list[int], list[float]]:
    MAXIMUM_NUMBER_OF_CUPS: int = 10
    ERROR_PERCENTAGE: int = 0.02

    number_of_cups = np.arange(1, MAXIMUM_NUMBER_OF_CUPS+1)

    mass = number_of_cups * experiment.base_mass_si
    times = falltime_with_air_resistance(mass, drag_coefficient, experiment.height_si, experiment.cross_section)

    return list(number_of_cups), list(times)


def _test() -> None:
    _HEIGHT     = 200 * METERS_PR_CENTIMITER
    _BASE_MASS  = 1 * KILOGRAMS_PR_GRAM
    _DIAMETER   = 2.0 * METERS_PR_CENTIMITER
    experiment = CupCakeExperiment(_HEIGHT, _BASE_MASS, _DIAMETER)

    _NUMBER_OF_CUPCAKES, _FALLTIME_IN_SECONDS = _secret_results(0.75, experiment)
    fit_drag_coeff, _ = calculate_drag_coefficient_from_fit(_NUMBER_OF_CUPCAKES, _FALLTIME_IN_SECONDS, experiment)

    draw_results(_NUMBER_OF_CUPCAKES, _FALLTIME_IN_SECONDS, experiment, fit_drag_coeff=fit_drag_coeff)
    draw_results(_NUMBER_OF_CUPCAKES, _FALLTIME_IN_SECONDS, experiment, loglog=True)

if __name__ == '__main__':
    
    _test()
    plt.show()