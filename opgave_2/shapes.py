from math import pi
from abc import ABC, abstractproperty


class Shape(ABC):
    drag_coefficient: float

    @abstractproperty
    def cross_area(self) -> float:
        ...

    @abstractproperty
    def volume(self) -> float:
        ...

    @property
    def charecteristic_length_scale(self) -> float:
        return self.volume / self.cross_area
    

    def mass(self, density: float) -> float:
        return self.volume * density
    

    
def positive_length_scales(func: callable) -> callable:
    """Decorator that checks every input for non-zero values"""
    def with_check_args(self, *args, **kwargs):
        """Check input *args of the function "func" before inputting it"""
        for i, value in enumerate(args):
            if value <= 0:
                raise ValueError(f"Input argument {i} of function {func.__name__} is either negative or zero")

        return func(self, *args, **kwargs)

    
    return with_check_args


class Ball(Shape):
    drag_coefficient: float = 0.47
    radius: float
    
    @positive_length_scales
    def __init__(self, radius: float):
        self.radius = radius

    @property
    def volume(self) -> float:
        return 4/3 * pi * self.radius**3
    
    @property
    def cross_area(self) -> float:
        return pi * self.radius**2


class HalfBall(Shape):
    radius: float

    @positive_length_scales
    def __init__(self, radius: float) -> None:
        self.radius = radius

    @property
    def cross_area(self) -> float:
        return pi ** self.radius**2
    
    @property
    def volume(self) -> float:
        return 2/3 * pi * self.radius**3


class Cone(Shape):
    drag_coefficient: float = 0.5
    radius: float
    height: float

    @positive_length_scales
    def __init__(self, radius: float, height: float) -> None:
        self.radius = radius
        self.height = height

    @property
    def cross_area(self) -> float:
        return pi * self.radius**2
    
    @property
    def volume(self) -> float:
        return pi/3 * self.radius**2 * self.height


class Cylinger(Shape):
    drag_coefficient: float = 0.82
    radius: float
    height: float

    @positive_length_scales
    def __init__(self, radius: float, height: float) -> None:
        self.radius = radius
        self.height = height

    @property
    def cross_area(self) -> float:
        return pi * self.radius**2
    
    @property
    def volume(self) -> float:
        return pi/3 * self.radius**2 * self.height
    

class Cube(Shape):
    drag_coefficient: float = 1.15
    side: float

    @positive_length_scales
    def __init__(self, side: float):
        self.side = side

    @property
    def cross_area(self) -> float:
        return self.side**2
    
    @property
    def volume(self) -> float:
        return self.side**3


class Rocket(Shape):
    """Souce of drag coefficient: https://www.grc.nasa.gov/www/k-12/rocket/shaped.html"""
    drag_coefficient: float = 0.75
    radius: float
    top_height: float
    body_height: float

    top: Shape
    body: Shape

    @positive_length_scales
    def __init__(self, radius: float, top_height: float, body_height: float) -> None:
        self.radius = radius
        self.top_height = top_height
        self.body_height = body_height

        self.top = Cone(radius, top_height)
        self.body = Cylinger(radius, body_height)


    @property
    def cross_area(self) -> float:
        return self.top.cross_area
    
    @property
    def volume(self) -> float:
        return self.top.volume + self.body.volume


class Bullet(Shape):
    """Souce of drag coefficient: https://www.grc.nasa.gov/www/k-12/rocket/shaped.html"""
    drag_coefficient: float = 0.295

    radius: float
    body_height: float

    top: Shape
    body: Shape

    @positive_length_scales
    def __init__(self, radius: float, body_height: float) -> None:
        self.radius = radius
        
        self.body_height = body_height
        self.top = HalfBall(radius)
        self.body = Cylinger(radius, body_height)


    @property
    def cross_area(self) -> float:
        return self.top.cross_area
    
    @property
    def volume(self) -> float:
        return self.top.volume + self.body.volume


if __name__ == '__main__':
    Ball(10).volume
    Rocket(10, 10, 10)