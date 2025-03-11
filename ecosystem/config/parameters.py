# config/parameters.py - Entity-specific parameters

# Plant parameters
PLANT_PARAMS = {
    'initial_energy': 100,
    'growth_rate_range': (0.01, 0.05),
    'max_size_range': (20, 40),
    'reproduction_threshold': 50,
    'reproduction_chance': 0.01,
    'reproduction_cost': 30,
    'energy_gain_rate': 1
}

# Sheep parameters
SHEEP_PARAMS = {
    'initial_energy': 120,
    'size': 25,
    'speed_range': (0.5, 2.0),
    'reproduction_threshold': 150,
    'reproduction_chance': 0.005,
    'reproduction_cost': 50,
    'energy_consumption_rate': 0.2
}

# Fox parameters
FOX_PARAMS = {
    'initial_energy': 160,
    'size': 30,
    'speed_range': (1.0, 3.0),
    'reproduction_threshold': 200,
    'reproduction_chance': 0.003,
    'reproduction_cost': 70,
    'energy_consumption_rate': 0.3
}