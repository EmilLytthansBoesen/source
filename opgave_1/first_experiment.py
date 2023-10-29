import numpy as np
import matplotlib.pyplot as plt

from matplotlib.axes import Axes

METERS_PR_CENTIMETER = 1 / 100
FONTSIZE = 12

RealArray = np.ndarray[float]


def main(height_cm: list[int], time_seconds: list[float]) -> None:
    height = METERS_PR_CENTIMETER * np.asarray(height_cm)
    time_seconds = np.asarray(time_seconds)

    fig, ax = plt.subplots(1, 2)

    draw_fall_curve(ax[0], time_seconds, height)
    draw_speed_curve(ax[1], time_seconds, height)


    fig.tight_layout()


def draw_fall_curve(ax: Axes, falltime: RealArray, height: RealArray) -> None:
    """Draw the distance moved by the object as a function of time

    Args:
        ax (Axes): _description_
        falltime (RealArray): _description_
        height (RealArray): _description_
    """
    prepare_plot(ax,
                 xlim=max(falltime),
                 ylim=max(height),
                 xlabel="Faldtid målt i sekunder", 
                 ylabel="Afstand bevæget mål i meter",
                 title="Faldkurve")
    ax.plot(falltime, height, '-*', color="purple")


def draw_speed_curve(ax: Axes, falltime: RealArray, height: RealArray) -> None:
    prepare_plot(ax,
                 xlim=max(falltime),
                 ylim=max(height),
                 xlabel="Faldtid målt i sekunder", 
                 ylabel="Gennemsnitsfart målt i m/s",
                 title="Fartkurve")
    
    ax.plot(falltime, height / falltime, '-*', color='purple')
    
    


def prepare_plot(ax: Axes, xlim:float=None, ylim:float=None, xlabel: str=None, ylabel: str=None, title: str=None) -> None:
    ax.grid()
    ax.axline((0,0), (0, 1), color='k')
    ax.axline((0,0), (1, 0), color='k')

    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.set_title(title, fontsize=FONTSIZE)

    if xlim is not None:
        ax.set_xlim((-0.25*xlim, xlim))
    if ylim is not None:
        ax.set_ylim((-0.25*ylim, ylim))



if __name__ == '__main__':
    main()
    plt.show()