"""
Module containing the functions used in the first exercise for the children
"""
import numpy as np
from matplotlib.pyplot import show

from opgave_1.first_experiment import main as draw_results_ball_drop
from opgave_1.mass_dependence import analyze_mass_dependence

__author__ = 'Emil Lytthans Boesen'



def main() -> None:
    """Run the complete first exercise with mocking input"""
    intro_question(skip=True)
    tegn_resultatet([...], [...], True)
    analyser_masse_afhængighed([...], [...], 0, True)

    show()



def intro_question(skip: bool=False) -> None:
    CURRENT_YEAR = 2023
    def born_in(age: int) -> int:   return CURRENT_YEAR - age
    if skip: 
        print("Remember to turn the skip option off!!!!!")
        return

    name: int = input("Hejsa. Hvad er dit navn? ")
    age: int  = int(input(f"Hejsa {name}! Hvor gammel er du? "))
    print(f"{name} er {age} år gammel og er født i år {born_in(age)}")



def tegn_resultatet(height: list[float], time: list[float], cheat: bool=False) -> None:
    """Draw the results the ball dropping experiment"""
    if cheat or ellipsis in height:
        height = np.arange(1, 20, 1)
        time = np.sqrt(height / 10)

    draw_results_ball_drop(height, time)
    print("Som i nok kan se fra dataen, flader fartkurven ud.")



def analyser_masse_afhængighed(number_of_forms: list[float], falltime_seconds: list[float], mass_single_form: float, cheat: bool=False) -> None:
    """Drop the results of the cupcake dropping experiment"""
    if cheat:
        number_of_forms = list(range(1, 21))
        falltime_seconds = 10 * np.sqrt(number_of_forms)
        mass_single_form = 1.0
    
    analyze_mass_dependence(number_of_forms, falltime_seconds, mass_single_form)
    




def at_import_error() -> None:
    print("Åhhh nej. Det ser ud til at Emil har fucket up. Hent ham lige og fortæl ham har han er en spade")


def at_unknown_error(error: Exception, as_test: bool=True) -> None:
    if not as_test:
        raise error
    
    print(f"Det ser ud til at der er {type(error)} - Hent lige Emil og lad ham hjælpe jer")


if __name__ == '__main__':
    main()