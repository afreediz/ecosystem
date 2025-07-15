# config/parameters.py - Entity-specific parameters
from pydantic import BaseModel
from dataclasses import dataclass
from typing import Tuple

@dataclass
class ENTITY_IN_PRECEPTION:
    empty = 0
    plant = 1
    sheep = 2
    fox = 3
    sheep_and_fox = 4
    sheep_and_plant = 5
    fox_and_plant = 6

# Plant parameters
@dataclass
class PLANT_PARAMS:
    initial_energy:int|float = 100
    growth_rate:float = 0.03
    max_size:float = 30
    reproduction_threshold:int = 50
    reproduction_chance:float = 0.01
    reproduction_cost:int =  30
    energy_gain_rate:int = 1


# Sheep parameters

@dataclass
class SHEEP_PARAMS:
    initial_energy: int|float = 120
    size:int = 25
    speed:float = 2
    reproduction_threshold:int = 150
    reproduction_chance:float = 0.005
    reproduction_cost:int = 50
    energy_consumption_rate:float = 0.2
    vision_range:int = 80

# Fox parameters
@dataclass
class FOX_PARAMS:
    initial_energy:int|float = 500
    size:int = 30
    speed:float = 3
    reproduction_threshold:int = 200
    reproduction_chance:float = 0.003
    reproduction_cost:int = 70
    energy_consumption_rate:float = 0.7
    vision_range:int = 120