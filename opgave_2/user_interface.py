import ipywidgets as widgets


# Measured in g/cm^3
MATERIAL_DENSITIES = {
    "Aluminium": 2.7,
    "Egetræ": 0.85,
    "Granit": 2.650,
    "Vand": 1.00,
    "Uran": 18.70,
    "Magnesium": 1.74,
    "Messing": 8.40,
    "Candy floss": 0.059,
}


MaterialDropDown = widgets.Dropdown(
    options=[(f"{name} ({value} g/cm^3)", value) for (name, value) in MATERIAL_DENSITIES.items()],
    description='Materiale: ',
)


AngleSlider = widgets.FloatSlider(
    value = 30,
    min = 0,
    max = 90,
    step = 0.1,
    description= "Vinkel",
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
)


LengthScaleSlider = widgets.FloatSlider(
    value = 10,
    min = 10,
    max = 100,
    step = 0.1,
    description= "Vinkel",
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
)