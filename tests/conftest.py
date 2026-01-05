"""
Pytest fixtures для тестов oscillators.
"""

import pytest
import numpy as np
from oscillators import (
    SpinType, ParticleType, Particle, QuantumOscillator,
    PhysicalConstants, ParametricResonance, LeptogenesisModel,
    QuantumCreationInExpandingUniverse, MatterGenesisSimulation,
    PrimordialOscillatorUniverse, DetailedMatterGenesis,
)


@pytest.fixture
def sample_particle():
    """Создание тестовой частицы."""
    return Particle(
        type=ParticleType.QUARK,
        energy=1.0,
        momentum=np.array([0.5, 0.5, 0.5]),
        position=np.array([0.0, 0.0, 0.0]),
        spin=0.5,
        creation_time=0.0,
        antiparticle=False
    )


@pytest.fixture
def sample_antiparticle():
    """Создание тестовой античастицы."""
    return Particle(
        type=ParticleType.QUARK,
        energy=1.0,
        momentum=np.array([-0.5, -0.5, -0.5]),
        position=np.array([0.1, 0.1, 0.1]),
        spin=0.5,
        creation_time=0.0,
        antiparticle=True
    )


@pytest.fixture
def sample_photon():
    """Создание тестового фотона."""
    return Particle(
        type=ParticleType.PHOTON,
        energy=1.0,
        momentum=np.array([1.0, 0.0, 0.0]),
        position=np.array([0.0, 0.0, 0.0]),
        spin=1.0,
        creation_time=0.0,
        antiparticle=False
    )


@pytest.fixture
def scalar_oscillator():
    """Создание скалярного осциллятора."""
    return QuantumOscillator(
        frequency=1.0,
        amplitude=1.0 + 0j,
        spin=SpinType.SCALAR,
        position=np.array([0.0, 0.0, 0.0])
    )


@pytest.fixture
def spinor_oscillator():
    """Создание спинорного осциллятора."""
    return QuantumOscillator(
        frequency=2.0,
        amplitude=0.5 + 0.5j,
        spin=SpinType.SPINOR,
        position=np.array([1.0, 1.0, 1.0])
    )


@pytest.fixture
def vector_oscillator():
    """Создание векторного осциллятора."""
    return QuantumOscillator(
        frequency=0.5,
        amplitude=2.0 + 0j,
        spin=SpinType.VECTOR,
        position=np.array([0.5, 0.5, 0.5])
    )


@pytest.fixture
def parametric_resonance():
    """Создание модели параметрического резонанса."""
    return ParametricResonance(inflaton_mass=1e13, coupling=1e-7)


@pytest.fixture
def leptogenesis_model():
    """Создание модели лептогенеза."""
    return LeptogenesisModel(M=1e10, Yukawa=1e-6, CP_violation=1e-10)


@pytest.fixture
def quantum_creation_model():
    """Создание модели квантового рождения."""
    return QuantumCreationInExpandingUniverse(mass=0.1, expansion_rate=0.01)


@pytest.fixture
def matter_genesis_simulation():
    """Создание симуляции рождения материи с минимальными параметрами."""
    return MatterGenesisSimulation(
        volume_size=1.0,
        initial_inflaton_energy=1e12,
        hubble_parameter=1e-5,
        reheating_temperature=1e9,
        cp_violation=1e-10
    )


@pytest.fixture
def primordial_universe():
    """Создание первичной вселенной осцилляторов."""
    return PrimordialOscillatorUniverse(total_energy=10.0, initial_symmetry=True)


@pytest.fixture
def detailed_genesis():
    """Создание детальной модели генезиса."""
    return DetailedMatterGenesis()


# Утилиты для тестов
@pytest.fixture
def random_seed():
    """Фиксирование seed для воспроизводимости."""
    np.random.seed(42)
    return 42


@pytest.fixture
def small_particle_list(sample_particle, sample_antiparticle, sample_photon):
    """Небольшой список частиц для тестов."""
    return [sample_particle, sample_antiparticle, sample_photon]

