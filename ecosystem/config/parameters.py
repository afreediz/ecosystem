# config/parameters.py - Entity-specific parameters
from pydantic import BaseModel
from dataclasses import dataclass
from typing import Tuple

# Plant parameters
@dataclass
class PLANT_PARAMS:
    initial_energy:int|float = 100
    growth_rate_range:Tuple[float, ...] = (0.01, 0.05)
    max_size_range:Tuple[int, ...] = (20, 40)
    reproduction_threshold:int = 50
    reproduction_chance:float = 0.01
    reproduction_cost:int =  30
    energy_gain_rate:int = 1


# Sheep parameters

@dataclass
class SHEEP_PARAMS:
    initial_energy: int|float = 120
    size:int = 25
    speed_range: Tuple[float, ...] = (1.5, 4.0)
    reproduction_threshold:int = 150
    reproduction_chance:float = 0.005
    reproduction_cost:int = 50
    energy_consumption_rate:float = 0.2
    max_age:int = 30

class SHEEP_VISUAL_PARAMS:
    pass

# Fox parameters
@dataclass
class FOX_PARAMS:
    initial_energy:int|float = 500
    size:int = 30
    speed_range: Tuple[float, ...] = (1.0, 3.0)
    reproduction_threshold:int = 200
    reproduction_chance:float = 0.003
    reproduction_cost:int = 70
    energy_consumption_rate:float = 0.7
    max_age:int = 50
