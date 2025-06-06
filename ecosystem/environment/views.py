from pydantic import BaseModel
from enum import Enum

class Climate(str, Enum):
    SUNNY = 'sunny'
    RAINING = 'raining'
    THUNDERSTORM = 'thunderstorm'

class Season(str, Enum):
    SPRING = 'spring'
    SUMMER = 'summer'
    WINTER = 'winter'
    AUTUMN = 'autumn'

class Statistics(BaseModel):
    plants:int
    sheep:int
    foxes:int
    season:Season
    climate:Climate
    frame:int
    day:int
    month:int
    year:int