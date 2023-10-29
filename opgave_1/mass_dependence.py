import numpy as np

if __name__ == '__main__':
    from fit_model import FitModel, InverseSquareRoot
else:
    from opgave_1.fit_model import FitModel, InverseSquareRoot

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from scipy.optimize import curve_fit
from typing import Callable
from functools import partial


KILOGRAMS_PR_GRAM: float = 1 / 1000
FONTSIZE: int = 12


def fit_model_plot_factory(number_of_forms: list[int], falltimes_second: list[float], single_form_mass_gram: float, *, cheat=True) -> Callable[[FitModel], None]:
    if cheat:
        number_of_forms = np.arange(1, 21)
        falltimes_second = 10 / np.sqrt(number_of_forms)
        single_form_mass_gram = 1.00
    
    
    def plot_model(model: FitModel) -> None:
        show_model_fit(model, number_of_forms, falltimes_second, single_form_mass_gram)
    return plot_model
    

def show_model_fit(model: FitModel, number_of_forms: list[int], falltimes_second: list[float], single_form_mass_gram: float) -> None:
    fig, ax = plt.subplots(1, 2)

    mass     = single_form_mass_gram * np.asarray(number_of_forms)
    falltime = np.asarray(falltimes_second)
    params = fit_data_to_model(model, mass, falltime)
    
    def fit(x: np.ndarray[float]) -> np.ndarray[float]:
        return model(x, *params)

    analyze_mass_dependence(number_of_forms, falltimes_second, single_form_mass_gram, ax=ax[0], fit=fit, loglog=False)
    analyze_mass_dependence(number_of_forms, falltimes_second, single_form_mass_gram, ax=ax[1], fit=fit, loglog=True)

    fig.tight_layout()


def analyze_mass_dependence(number_of_forms: list[int], 
                            falltimes_second: list[float], 
                            single_form_mass_gram: float,
                            *,
                            ax: Axes=None,
                            fit: Callable[[np.ndarray[float]], np.ndarray[float]]=None,
                            loglog: bool=False) -> None:
    """Plot the data and find the best fit to the c / sqrt(m) law.

    Args:
        number_of_forms (list[int]): Number of cake forms 
        falltimes_second (int[float]): Falltime for each experiment
    """
    
    if (number_of_forms is Ellipsis) or (falltimes_second is None) or (single_form_mass_gram is None):
        print("Hov! I har vist ikke indtast alle de nødvendige datapunkter")
        return
    if len(number_of_forms) != len(falltimes_second):
        print("Antallet af datapunkter matcher ikke!")
        return
    
    mass = single_form_mass_gram * np.asarray(number_of_forms)
    falltimes_second = np.asarray(falltimes_second)

    if ax is None:  _, ax = plt.subplots()

    prepare_plot(ax, x_limit=max(mass), loglog=loglog)
    if fit is not None: plot_fit(ax, mass, fit)
    plot_datapoints(ax, mass, falltimes_second)
    

def fit_data_to_model(model: FitModel, xdata: np.ndarray[float], ydata: np.ndarray[float]) -> tuple[list[float], list[float]]:
    params, _ = curve_fit(model.__call__, xdata, ydata)
    return list(params)
        
        
def prepare_plot(ax: Axes, *, x_limit: float=None, y_limit: float=None, loglog: bool=False) -> None:
    """Prepare the plot by setting the axis limits, draw coordinate system and set the labels and title"""
    ax.grid()
    ax.axline((0, 0), (0, 1), color='k')  # x-axis
    ax.axline((0, 0), (1, 0), color='k')  # y-axis
    
    ax.set_title("Faldtiders afhængighed af masse", fontsize=FONTSIZE)
    ax.set_xlabel("Masse mål i gram", fontsize=FONTSIZE)
    ax.set_ylabel("Faldtid målt i sekunder", fontsize=FONTSIZE)

    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
        return

    if x_limit is not None:
        ax.set_xlim((-0.25*x_limit, x_limit))
    if y_limit is not None:
        ax.set_ylim((-0.25*y_limit, y_limit))


def plot_fit(ax: Axes, mass: np.ndarray[float], fit: Callable[[np.ndarray[float]], np.ndarray[float]]) -> None:
    continuous_mass = np.linspace(min(mass), max(mass))
    ax.plot(continuous_mass, fit(continuous_mass), '-', color="purple", label="Den bedste linje")


def plot_datapoints(ax: Axes, mass: np.ndarray[float], falltime: np.ndarray[float]) -> None:
    ax.plot(mass, falltime, '*', color="purple")


def _test(*args, **kwargs) -> None:
    from math import pi, sqrt
    base_mass = 0.01
    number_of_forms = list(range(1, 21))
    falltimes = [pi / sqrt(n) for n in number_of_forms]
    
    fit_model_plot_factory(number_of_forms=number_of_forms, falltimes_second=falltimes, single_form_mass_gram=base_mass)(InverseSquareRoot())
    plt.show()


if __name__ == '__main__':
    _test()