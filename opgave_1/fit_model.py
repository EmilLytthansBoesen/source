import numpy as np
from numpy import ndarray
from typing import Any

class FitModel:
    name: str
    _as_string: str

    def __call__(self, x: ndarray[float], **params: float) -> ndarray[float]:
        raise NotImplemented("The __call__ method is not implemented")
    
    def __str__(self) -> str:
        return f"{self.name.capitalize()}: ({self._as_string})"

    def __repr__(self) -> str:
        return self.__str__()
    

class LinearModel(FitModel):
    name: str = 'Lineær'
    _as_string: str = "a*x + b"
    
    def __call__(self, x: ndarray[float, Any], a: float, b: float) -> ndarray[float, Any]:
        return a*x + b
    

class Exponential(FitModel):
    name: str = "exponentiel"
    _as_string: str = "a * exp(-b*x)"

    def __call__(self, x: ndarray[float, Any], a: float, b: float) -> ndarray[float, Any]:
        return a * np.exp(-b * x)

class SecondOrderPolynomial(FitModel):
    name = "andengrads polynomium"
    _as_string = "a*x^2 + b*x +c"

    def __call__(self, x: ndarray[float, Any], a: float, b: float, c: float) -> ndarray[float, Any]:
        return a*x**2 + b*x + c


class ThirdOrderPolynomial(FitModel):
    name = "tredjegrads polynomium"
    _as_string = "a*x^3 + b*x^2 + c*x + d"

    def __call__(self, x: ndarray[float, Any], a: float, b: float, c: float, d: float) -> ndarray[float, Any]:
        return a*x**3 + b*x**2 + c*x + d


class InverseSquareRoot(FitModel):
    name = "invers kvadratrot"
    _as_string = "a / sqrt(x)"

    def __call__(self, x: ndarray[float, Any], a: float) -> ndarray[float, Any]:
        return a / np.sqrt(x)
    


FIT_MODELS: list[FitModel] = [
    LinearModel(), 
    SecondOrderPolynomial(),
    ThirdOrderPolynomial(),
    InverseSquareRoot(),
    Exponential()
]


if __name__ == '__main__':
    
    for model in FIT_MODELS:
        print(model)
