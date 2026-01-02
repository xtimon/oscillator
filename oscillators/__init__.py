"""
Oscillators - библиотека для моделирования рождения материи во Вселенной.

Эта библиотека реализует различные космологические модели:
- Параметрический резонанс и разогрев после инфляции
- Лептогенез и барионная асимметрия
- Квантовое рождение частиц в расширяющейся Вселенной
- Симуляция полного процесса genesis материи

Основные классы:
-----------------
Базовые типы (core):
    - SpinType: типы спина частиц (скаляр, спинор, вектор, тензор)
    - ParticleType: типы частиц (кварки, лептоны, фотоны и др.)
    - Particle: класс частицы с энергией, импульсом и положением
    - QuantumOscillator: квантовый осциллятор с учетом спина

Физические модели (models):
    - ParametricResonance: параметрический резонанс при разогреве
    - LeptogenesisModel: генерация барионной асимметрии
    - QuantumCreationInExpandingUniverse: рождение из вакуума

Симуляции (simulation):
    - MatterGenesisSimulation: рождение материи из инфлатона
    - PrimordialOscillatorUniverse: модель осцилляторов
    - DetailedMatterGenesis: полная интегрированная модель

Быстрый старт:
--------------
>>> from oscillators import MatterGenesisSimulation
>>> sim = MatterGenesisSimulation(volume_size=10.0)
>>> history = sim.evolve_universe(total_time=1000.0)
>>> sim.visualize_genesis(history)

Версия: 1.0.0
Автор: Timur Isanov <xtimon@yahoo.com>
"""

__version__ = "0.1.0"
__author__ = "Timur Isanov <xtimon@yahoo.com>"

# Базовые типы данных
from .core import (
    SpinType,
    ParticleType,
    Particle,
    QuantumOscillator,
    PhysicalConstants,
    get_particle_mass,
    get_particle_spin,
)

# Физические модели
from .models import (
    ParametricResonance,
    LeptogenesisModel,
    QuantumCreationInExpandingUniverse,
)

# Симуляции
from .simulation import (
    MatterGenesisSimulation,
    PrimordialOscillatorUniverse,
    DetailedMatterGenesis,
)

# Публичный API
__all__ = [
    # Версия
    "__version__",
    "__author__",
    
    # Базовые типы
    "SpinType",
    "ParticleType", 
    "Particle",
    "QuantumOscillator",
    "PhysicalConstants",
    "get_particle_mass",
    "get_particle_spin",
    
    # Модели
    "ParametricResonance",
    "LeptogenesisModel",
    "QuantumCreationInExpandingUniverse",
    
    # Симуляции
    "MatterGenesisSimulation",
    "PrimordialOscillatorUniverse",
    "DetailedMatterGenesis",
]


def info():
    """Выводит информацию о библиотеке."""
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                   OSCILLATORS LIBRARY v{__version__}                     ║
╠══════════════════════════════════════════════════════════════════╣
║  Библиотека для моделирования рождения материи во Вселенной      ║
╠══════════════════════════════════════════════════════════════════╣
║  Доступные модули:                                               ║
║    • core       - базовые типы данных и частицы                  ║
║    • models     - физические модели (резонанс, лептогенез)       ║
║    • simulation - комплексные симуляции                          ║
╠══════════════════════════════════════════════════════════════════╣
║  Примеры запуска: python -m oscillators.examples                 ║
╚══════════════════════════════════════════════════════════════════╝
""")
