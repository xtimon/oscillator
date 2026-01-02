"""
Классы комплексных симуляций космологических процессов.

Содержит:
- MatterGenesisSimulation: симуляция рождения материи из инфлатона
- PrimordialOscillatorUniverse: модель Вселенной как осцилляторов
- DetailedMatterGenesis: полная интегрированная модель
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .core import (
    SpinType, ParticleType, Particle, QuantumOscillator,
    get_particle_mass, get_particle_spin
)
from .models import ParametricResonance, LeptogenesisModel, QuantumCreationInExpandingUniverse


class MatterGenesisSimulation:
    """
    Симуляция рождения материи из первичных флуктуаций.
    
    Моделирует полный процесс от инфляции до формирования
    частиц Стандартной Модели, включая:
    - Эволюцию поля инфлатона
    - Параметрический резонанс (разогрев)
    - Рождение частиц и античастиц
    - CP-нарушение и барионную асимметрию
    - Аннигиляцию частиц
    
    Attributes:
        volume_size: размер моделируемого объема
        time: текущее время симуляции
        scale_factor: масштабный фактор Вселенной
        particles: список рожденных частиц
        temperature: температура (GeV)
    
    Example:
        >>> sim = MatterGenesisSimulation(volume_size=10.0)
        >>> history = sim.evolve_universe(total_time=1000.0, dt=0.1)
        >>> sim.visualize_genesis(history)
    """
    
    def __init__(
        self, 
        volume_size: float = 1.0,
        initial_inflaton_energy: float = 1e16,
        hubble_parameter: float = 1e-5
    ):
        """
        Инициализация симуляции.
        
        Args:
            volume_size: начальный размер объема
            initial_inflaton_energy: энергия инфлатона (GeV)
            hubble_parameter: начальный параметр Хаббла
        """
        self.volume_size = volume_size
        self.time = 0.0
        self.scale_factor = 1e-30
        
        # Параметры инфлатона
        self.inflaton_field = initial_inflaton_energy
        self.inflaton_velocity = 0.0
        self.inflaton_mass = 1e13  # GeV
        
        # Параметры расширения
        self.hubble = hubble_parameter
        self.expansion_rate = hubble_parameter
        
        # Частицы
        self.particles: List[Particle] = []
        self.particle_statistics = {ptype: 0 for ptype in ParticleType}
        
        # CP-нарушение
        self.CP_violation_parameter = 1e-8
        self.baryon_asymmetry = 0.0
        
        # Температура
        self.temperature = 1e15  # GeV
        
    def inflaton_potential(self, phi: float) -> float:
        """
        Потенциал инфлатона V(φ) = ½m²φ² + ¼λφ⁴
        
        Args:
            phi: значение поля инфлатона
            
        Returns:
            значение потенциала V(φ)
        """
        # Защита от переполнения
        phi_clipped = np.clip(phi, -1e20, 1e20)
        term1 = 0.5 * self.inflaton_mass**2 * phi_clipped**2
        term2 = 0.25 * 1e-14 * phi_clipped**4
        return np.clip(term1 + term2, 0, 1e40)
    
    def inflaton_potential_derivative(self, phi: float) -> float:
        """Производная потенциала dV/dφ"""
        phi_clipped = np.clip(phi, -1e20, 1e20)
        term1 = self.inflaton_mass**2 * phi_clipped
        term2 = 1e-14 * phi_clipped**3
        return np.clip(term1 + term2, -1e40, 1e40)
    
    def evolve_inflaton(self, dt: float) -> float:
        """
        Эволюция поля инфлатона.
        
        Решает уравнение Клейна-Гордона:
        φ̈ + 3Hφ̇ + dV/dφ = 0
        
        Args:
            dt: временной шаг
            
        Returns:
            плотность энергии инфлатона
        """
        # Трение из-за расширения
        friction = 3 * self.hubble * self.inflaton_velocity
        
        # Ускорение
        acceleration = -friction - self.inflaton_potential_derivative(self.inflaton_field)
        
        # Ограничение ускорения
        acceleration = np.clip(acceleration, -1e20, 1e20)
        
        # Обновление
        self.inflaton_velocity += acceleration * dt
        self.inflaton_velocity = np.clip(self.inflaton_velocity, -1e20, 1e20)
        
        self.inflaton_field += self.inflaton_velocity * dt
        self.inflaton_field = np.clip(self.inflaton_field, -1e20, 1e20)
        
        # Плотность энергии
        energy_density = (
            0.5 * self.inflaton_velocity**2 + 
            self.inflaton_potential(self.inflaton_field)
        )
        
        return min(energy_density, 1e40) if np.isfinite(energy_density) else 0
    
    def parametric_resonance(self, time: float) -> float:
        """
        Фактор параметрического резонанса.
        
        Args:
            time: текущее время
            
        Returns:
            инкремент неустойчивости
        """
        omega_inflaton = self.inflaton_mass
        phi_amplitude = min(np.abs(self.inflaton_field), 1e20)
        
        # Защита от переполнения
        if self.inflaton_mass > 0:
            q = min(phi_amplitude**2 / (4 * self.inflaton_mass**2), 100)
        else:
            q = 0
        
        if q > 0:
            exponent = min(0.5 * q * omega_inflaton * time, 50)
            return np.exp(exponent)
        return 1.0
    
    def _get_particle_probabilities(self) -> Dict[ParticleType, float]:
        """Вероятности рождения частиц разных типов."""
        if self.temperature > 1e14:
            return {
                ParticleType.INFLATON: 0.4,
                ParticleType.HIGGS: 0.3,
                ParticleType.PHOTON: 0.2,
                ParticleType.QUARK: 0.05,
                ParticleType.LEPTON: 0.05,
                ParticleType.DARK_MATTER: 0.0
            }
        elif self.temperature > 1e3:
            return {
                ParticleType.INFLATON: 0.1,
                ParticleType.HIGGS: 0.2,
                ParticleType.PHOTON: 0.3,
                ParticleType.QUARK: 0.2,
                ParticleType.LEPTON: 0.15,
                ParticleType.DARK_MATTER: 0.05
            }
        else:
            return {
                ParticleType.INFLATON: 0.01,
                ParticleType.HIGGS: 0.04,
                ParticleType.PHOTON: 0.4,
                ParticleType.QUARK: 0.2,
                ParticleType.LEPTON: 0.2,
                ParticleType.DARK_MATTER: 0.15
            }
    
    def _determine_antiparticle(self, ptype: ParticleType) -> bool:
        """Определение частица/античастица с CP-нарушением."""
        if ptype in [ParticleType.QUARK, ParticleType.LEPTON]:
            p_antiparticle = 0.5 - self.CP_violation_parameter
            return np.random.random() > p_antiparticle
        return False
    
    def create_particles_from_inflaton(self, dt: float) -> List[Particle]:
        """
        Рождение частиц из распада инфлатона.
        
        Args:
            dt: временной шаг
            
        Returns:
            список новых частиц
        """
        available_energy = abs(self.inflaton_velocity) * abs(self.inflaton_field) * dt
        
        # Защита от inf и nan
        if not np.isfinite(available_energy) or available_energy <= 0:
            return []
        
        # Ограничиваем доступную энергию
        available_energy = min(available_energy, 1e30)
        
        new_particles = []
        particle_probabilities = self._get_particle_probabilities()
        
        # Ограничиваем число частиц для стабильности
        n_particles_possible = min(int(available_energy / 100.0), 1000)
        lam = min(n_particles_possible * 0.1, 100)  # Ограничение для Пуассона
        n_particles = np.random.poisson(lam) if lam > 0 else 0
        
        for _ in range(n_particles):
            ptype = np.random.choice(
                list(particle_probabilities.keys()),
                p=list(particle_probabilities.values())
            )
            
            mass = get_particle_mass(ptype)
            
            if mass > available_energy / max(1, n_particles):
                continue
            
            energy = mass + np.random.random() * (available_energy/n_particles - mass)
            momentum_mag = np.sqrt(max(0, energy**2 - mass**2))
            
            # Случайное направление
            theta = np.random.random() * 2 * np.pi
            phi = np.random.random() * np.pi
            momentum = np.array([
                momentum_mag * np.sin(phi) * np.cos(theta),
                momentum_mag * np.sin(phi) * np.sin(theta),
                momentum_mag * np.cos(phi)
            ])
            
            position = np.random.random(3) * self.volume_size
            spin = get_particle_spin(ptype)
            is_antiparticle = self._determine_antiparticle(ptype)
            
            particle = Particle(
                type=ptype,
                energy=energy,
                momentum=momentum,
                position=position,
                spin=spin,
                creation_time=self.time,
                antiparticle=is_antiparticle
            )
            
            new_particles.append(particle)
            self.particle_statistics[ptype] += 1
            
            if is_antiparticle:
                self.baryon_asymmetry -= 1
            else:
                self.baryon_asymmetry += 1
        
        return new_particles
    
    def _annihilate_particles(self, dt: float):
        """Аннигиляция частиц и античастиц."""
        if len(self.particles) < 2:
            return
        
        to_remove = set()
        
        for i in range(len(self.particles)):
            if i in to_remove:
                continue
                
            pi = self.particles[i]
            
            if pi.type not in [ParticleType.QUARK, ParticleType.LEPTON]:
                continue
            
            for j in range(i+1, len(self.particles)):
                if j in to_remove:
                    continue
                    
                pj = self.particles[j]
                
                if (pi.type == pj.type and pi.antiparticle != pj.antiparticle):
                    dist = np.linalg.norm(pi.position - pj.position)
                    annihilation_prob = np.exp(-dist * 10.0) * dt * 0.1
                    
                    if np.random.random() < annihilation_prob:
                        to_remove.add(i)
                        to_remove.add(j)
                        
                        n_photons = np.random.poisson(2)
                        
                        for _ in range(n_photons):
                            photon_energy = (pi.energy + pj.energy) / max(1, n_photons)
                            photon = Particle(
                                type=ParticleType.PHOTON,
                                energy=photon_energy,
                                momentum=np.random.randn(3) * photon_energy,
                                position=(pi.position + pj.position) / 2,
                                spin=1,
                                creation_time=self.time
                            )
                            self.particles.append(photon)
                            self.particle_statistics[ParticleType.PHOTON] += 1
                        
                        break
        
        self.particles = [p for i, p in enumerate(self.particles) if i not in to_remove]
    
    def evolve_universe(
        self, 
        total_time: float = 1000.0, 
        dt: float = 0.1,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Основная эволюция Вселенной.
        
        Args:
            total_time: полное время симуляции
            dt: временной шаг
            show_progress: показывать прогресс-бар
            
        Returns:
            история симуляции
        """
        time_steps = int(total_time / dt)
        history = []
        
        iterator = tqdm(range(time_steps), desc="Симуляция") if show_progress else range(time_steps)
        
        for step in iterator:
            self.time += dt
            
            # 1. Расширение (с защитой от переполнения)
            expansion = np.exp(min(self.hubble * dt, 10))
            self.scale_factor = min(self.scale_factor * expansion, 1e30)
            self.volume_size = min(self.volume_size * expansion**3, 1e90)
            
            # 2. Остывание (с защитой)
            if self.scale_factor > 1e-30:
                self.temperature = max(self.temperature / expansion, 1e-20)
            
            # 3. Эволюция инфлатона
            inflaton_energy = self.evolve_inflaton(dt)
            
            # 4. Резонанс
            resonance_factor = self.parametric_resonance(self.time)
            
            # 5. Рождение частиц
            if resonance_factor > 1.1 and inflaton_energy > 100.0 and np.isfinite(inflaton_energy):
                new_particles = self.create_particles_from_inflaton(dt)
                self.particles.extend(new_particles)
                
                total_new_energy = sum(p.energy for p in new_particles)
                if inflaton_energy > 0 and np.isfinite(total_new_energy):
                    energy_factor = min(total_new_energy / inflaton_energy, 0.9)
                    self.inflaton_field *= (1.0 - energy_factor * 0.1)
            
            # 6. Аннигиляция
            self._annihilate_particles(dt)
            
            # 7. Запись истории
            if step % 10 == 0:
                snapshot = {
                    'time': self.time,
                    'scale_factor': self.scale_factor,
                    'temperature': self.temperature,
                    'inflaton_energy': inflaton_energy,
                    'n_particles': len(self.particles),
                    'particle_stats': self.particle_statistics.copy(),
                    'baryon_asymmetry': self.baryon_asymmetry,
                    'resonance_factor': resonance_factor
                }
                history.append(snapshot)
                
                # Ограничение числа частиц
                if len(self.particles) > 10000:
                    self.particles = self.particles[-5000:]
        
        return history
    
    def visualize_genesis(self, history: List[Dict]):
        """
        Визуализация процесса рождения материи.
        
        Args:
            history: история симуляции
        """
        fig = plt.figure(figsize=(18, 12))
        
        times = [h['time'] for h in history]
        
        # 1. Температура и масштаб
        ax1 = plt.subplot(3, 3, 1)
        temps = [h['temperature'] for h in history]
        scale_factors = [h['scale_factor'] for h in history]
        
        ax1.semilogy(times, temps, 'r-', linewidth=2, label='Температура')
        ax1.set_xlabel('Время')
        ax1.set_ylabel('T (GeV)', color='r')
        ax1.tick_params(axis='y', labelcolor='r')
        ax1.grid(True, alpha=0.3)
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(times, scale_factors, 'b--', linewidth=2)
        ax1_twin.set_ylabel('a(t)', color='b')
        ax1_twin.tick_params(axis='y', labelcolor='b')
        ax1.set_title('Расширение и остывание')
        
        # 2. Энергия инфлатона
        ax2 = plt.subplot(3, 3, 2)
        inflaton_energies = [h['inflaton_energy'] for h in history]
        resonance_factors = [h['resonance_factor'] for h in history]
        
        ax2.plot(times, inflaton_energies, 'g-', linewidth=2)
        ax2.set_xlabel('Время')
        ax2.set_ylabel('Энергия инфлатона', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.grid(True, alpha=0.3)
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(times, resonance_factors, 'm:', linewidth=2)
        ax2_twin.set_ylabel('Резонанс', color='m')
        ax2_twin.tick_params(axis='y', labelcolor='m')
        ax2.set_title('Распад инфлатона')
        
        # 3. Количество частиц
        ax3 = plt.subplot(3, 3, 3)
        n_particles = [h['n_particles'] for h in history]
        ax3.plot(times, n_particles, 'k-', linewidth=2)
        ax3.set_xlabel('Время')
        ax3.set_ylabel('Количество частиц')
        ax3.set_title('Рождение материи')
        ax3.grid(True, alpha=0.3)
        
        # 4. Типы частиц
        ax4 = plt.subplot(3, 3, 4)
        colors = plt.cm.tab10(np.linspace(0, 1, len(ParticleType)))
        
        for i, ptype in enumerate(ParticleType):
            counts = [h['particle_stats'][ptype] for h in history]
            ax4.plot(times, counts, color=colors[i], label=ptype.value, linewidth=2)
        
        ax4.set_xlabel('Время')
        ax4.set_ylabel('Количество')
        ax4.set_title('Эволюция состава')
        ax4.legend(loc='best', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. Барионная асимметрия
        ax5 = plt.subplot(3, 3, 5)
        baryon_asymmetry = [h['baryon_asymmetry'] for h in history]
        ax5.plot(times, baryon_asymmetry, 'b-', linewidth=2)
        ax5.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax5.set_xlabel('Время')
        ax5.set_ylabel('Барионная асимметрия')
        ax5.set_title('CP-нарушение')
        ax5.grid(True, alpha=0.3)
        
        # 6. Космологические эпохи
        ax6 = plt.subplot(3, 3, 6)
        phases = [
            (1e-10, 1e15, "Инфляция", "#FF6B6B"),
            (1e-10, 1e14, "ВО", "#4ECDC4"),
            (1e-12, 1e3, "Электрослаб.", "#45B7D1"),
            (1e-5, 0.1, "КХД", "#96CEB4"),
        ]
        
        ax6.set_xscale('log')
        ax6.set_yscale('log')
        
        for time_start, temp, label, color in phases:
            ax6.fill_betweenx([temp*0.5, temp*2], time_start, time_start*100, 
                             alpha=0.3, color=color)
            ax6.text(time_start*10, temp, label, rotation=45, fontsize=8)
        
        ax6.set_xlabel('Время (log)')
        ax6.set_ylabel('Температура (GeV)')
        ax6.set_title('Космологические эпохи')
        ax6.grid(True, alpha=0.3)
        
        # 7. Энергетический спектр
        ax7 = plt.subplot(3, 3, 7)
        bins = np.logspace(-3, 6, 30)
        
        for ptype in ParticleType:
            energies = [p.energy for p in self.particles if p.type == ptype]
            if energies:
                ax7.hist(energies, bins=bins, alpha=0.5, label=ptype.value, density=True)
        
        ax7.set_xscale('log')
        ax7.set_xlabel('Энергия (GeV)')
        ax7.set_ylabel('Плотность')
        ax7.set_title('Энергетический спектр')
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3)
        
        # 8. Соотношение материя/антиматерия
        ax8 = plt.subplot(3, 3, 8)
        ratios, labels = [], []
        
        for ptype in [ParticleType.QUARK, ParticleType.LEPTON]:
            particles = [p for p in self.particles if p.type == ptype]
            if particles:
                n_p = sum(1 for p in particles if not p.antiparticle)
                n_anti = sum(1 for p in particles if p.antiparticle)
                
                ratio = n_p / n_anti if n_anti > 0 else float('inf')
                if not np.isinf(ratio):
                    ratios.append(ratio)
                    labels.append(ptype.value)
        
        if ratios:
            bars = ax8.bar(range(len(ratios)), ratios, alpha=0.7)
            ax8.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
            ax8.set_xticks(range(len(labels)))
            ax8.set_xticklabels(labels)
            ax8.set_ylabel('Частицы / Античастицы')
            ax8.set_title('Асимметрия материи')
            ax8.grid(True, alpha=0.3)
        
        # 9. 3D распределение
        ax9 = plt.subplot(3, 3, 9, projection='3d')
        
        if len(self.particles) > 0:
            sample_size = min(200, len(self.particles))
            sample_indices = np.random.choice(len(self.particles), size=sample_size, replace=False)
            
            colors_map = {
                ParticleType.INFLATON: 'red',
                ParticleType.PHOTON: 'yellow',
                ParticleType.QUARK: 'blue',
                ParticleType.LEPTON: 'green',
                ParticleType.DARK_MATTER: 'purple',
                ParticleType.HIGGS: 'orange'
            }
            
            for idx in sample_indices:
                p = self.particles[idx]
                color = colors_map.get(p.type, 'gray')
                marker = 'o' if not p.antiparticle else 'x'
                size = 20 + np.log10(max(1e-10, p.energy))
                ax9.scatter(p.position[0], p.position[1], p.position[2],
                           color=color, marker=marker, s=size, alpha=0.6)
        
        ax9.set_xlabel('X')
        ax9.set_ylabel('Y')
        ax9.set_zlabel('Z')
        ax9.set_title('Пространственное распределение')
        
        plt.tight_layout()
        plt.show()
        
        self._print_statistics(history[-1])
    
    def _print_statistics(self, final_snapshot: Dict):
        """Вывод итоговой статистики."""
        print("\n" + "="*60)
        print("ИТОГОВАЯ СТАТИСТИКА")
        print("="*60)
        
        print(f"\nВремя симуляции: {final_snapshot['time']:.2e}")
        print(f"Масштабный фактор: {final_snapshot['scale_factor']:.2e}")
        print(f"Температура: {final_snapshot['temperature']:.2e} GeV")
        print(f"Количество частиц: {final_snapshot['n_particles']}")
        print(f"Барионная асимметрия: {final_snapshot['baryon_asymmetry']:.2e}")
        
        print("\nРаспределение по типам:")
        total = sum(final_snapshot['particle_stats'].values())
        for ptype, count in final_snapshot['particle_stats'].items():
            pct = count / total * 100 if total > 0 else 0
            print(f"  {ptype.value:15s}: {count:8d} ({pct:6.2f}%)")
        
        n_baryons = sum(1 for p in self.particles 
                       if p.type in [ParticleType.QUARK, ParticleType.LEPTON] 
                       and not p.antiparticle)
        n_photons = sum(1 for p in self.particles if p.type == ParticleType.PHOTON)
        
        if n_photons > 0:
            baryon_to_photon = n_baryons / n_photons
            print(f"\nОтношение барионов к фотонам: {baryon_to_photon:.2e}")
            print(f"Наблюдаемое значение: ~6×10⁻¹⁰")
        
        print("="*60)


class PrimordialOscillatorUniverse:
    """
    Модель Вселенной как системы квантовых осцилляторов.
    
    Моделирует спонтанное нарушение симметрии при остывании,
    когда из единого первичного поля возникают частицы
    с различными спинами.
    
    Attributes:
        total_energy: полная энергия системы
        oscillators: список осцилляторов
        spin_distribution: распределение по спинам
    
    Example:
        >>> universe = PrimordialOscillatorUniverse(total_energy=50.0)
        >>> history = universe.simulate_symmetry_breaking(steps=500)
    """
    
    def __init__(self, total_energy: float = 100.0, initial_symmetry: bool = True):
        """
        Args:
            total_energy: полная энергия системы
            initial_symmetry: начинать с симметричного состояния
        """
        self.total_energy = total_energy
        self.oscillators: List[QuantumOscillator] = []
        self.spin_distribution = {s: 0 for s in SpinType}
        
        if initial_symmetry:
            self._create_symmetric_beginning()
    
    def _create_symmetric_beginning(self):
        """Создание первичного суперпозиционного состояния."""
        n_oscillators = 100
        energy_per_osc = self.total_energy / n_oscillators
        
        # Вероятности разных типов частиц
        spin_weights = {
            SpinType.SCALAR: 0.1,
            SpinType.SPINOR: 0.45,
            SpinType.VECTOR: 0.35,
            SpinType.TENSOR: 0.1
        }
        
        for i in range(n_oscillators):
            spin_type = np.random.choice(
                list(spin_weights.keys()),
                p=list(spin_weights.values())
            )
            
            freq_base = energy_per_osc * (1 + 0.1 * np.random.randn())
            
            # Разные частоты для разных спинов
            freq_factors = {
                SpinType.SCALAR: 0.8,
                SpinType.SPINOR: 1.2,
                SpinType.VECTOR: 1.0,
                SpinType.TENSOR: 0.5
            }
            freq = freq_base * freq_factors[spin_type]
            
            amplitude = np.exp(1j * np.random.random() * 2 * np.pi)
            amplitude *= np.sqrt(energy_per_osc)
            
            osc = QuantumOscillator(
                frequency=freq,
                amplitude=amplitude,
                spin=spin_type,
                position=np.random.randn(3) * 0.1
            )
            
            self.oscillators.append(osc)
            self.spin_distribution[spin_type] += 1
    
    def _interaction_probability(
        self, 
        spin1: SpinType, 
        spin2: SpinType, 
        temperature: float
    ) -> float:
        """Вероятность взаимодействия в зависимости от спинов."""
        base_prob = 0.01 * temperature
        
        # Принцип Паули для фермионов
        if spin1 == SpinType.SPINOR and spin2 == SpinType.SPINOR:
            return base_prob * 0.5
        
        # Сильное взаимодействие векторных бозонов
        elif spin1 == SpinType.VECTOR and spin2 == SpinType.VECTOR:
            return base_prob * 2.0
        
        # Слабое гравитационное взаимодействие
        elif spin1 == SpinType.TENSOR or spin2 == SpinType.TENSOR:
            return base_prob * 0.1
        
        return base_prob
    
    def _simulate_interaction(
        self, 
        osc1: QuantumOscillator, 
        osc2: QuantumOscillator, 
        temperature: float
    ):
        """Моделирование взаимодействия с возможным рождением частиц."""
        if np.random.random() < 0.001 * temperature:
            total_energy = np.abs(osc1.amplitude)**2 + np.abs(osc2.amplitude)**2
            
            if total_energy > 1.0:
                new_spin = self._generate_new_spin(osc1.spin, osc2.spin)
                new_freq = np.sqrt(total_energy) * 0.5
                
                new_osc = QuantumOscillator(
                    frequency=new_freq,
                    amplitude=np.sqrt(total_energy/2) * np.exp(1j*np.random.random()),
                    spin=new_spin,
                    position=(osc1.position + osc2.position)/2
                )
                
                self.oscillators.append(new_osc)
                self.spin_distribution[new_spin] += 1
    
    def _generate_new_spin(self, spin1: SpinType, spin2: SpinType) -> SpinType:
        """Правила рождения частиц с сохранением момента."""
        # Спин 1/2 + спин 1/2 = спин 0 или 1
        if spin1 == SpinType.SPINOR and spin2 == SpinType.SPINOR:
            return np.random.choice(
                [SpinType.SCALAR, SpinType.VECTOR], 
                p=[0.3, 0.7]
            )
        
        # Вектор + вектор может дать тензор
        elif spin1 == SpinType.VECTOR and spin2 == SpinType.VECTOR:
            if np.random.random() < 0.01:
                return SpinType.TENSOR
            return SpinType.VECTOR
        
        return np.random.choice([spin1, spin2])
    
    def simulate_symmetry_breaking(
        self, 
        temperature: float = 1.0, 
        steps: int = 1000,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Моделирование спонтанного нарушения симметрии.
        
        Args:
            temperature: начальная температура
            steps: число шагов
            show_progress: показывать прогресс
            
        Returns:
            история эволюции
        """
        print("Начальное распределение спинов:")
        for spin, count in self.spin_distribution.items():
            print(f"  {spin.name}: {count}")
        
        history = []
        iterator = tqdm(range(steps), desc="Эволюция") if show_progress else range(steps)
        
        for step in iterator:
            current_temp = temperature * np.exp(-step / 200)
            
            for i, osc1 in enumerate(self.oscillators):
                for j, osc2 in enumerate(self.oscillators[i+1:], i+1):
                    p_interact = self._interaction_probability(
                        osc1.spin, osc2.spin, current_temp
                    )
                    
                    if np.random.random() < p_interact:
                        self._simulate_interaction(osc1, osc2, current_temp)
            
            stats = {
                'step': step,
                'temperature': current_temp,
                'spin_counts': {s.name: 0 for s in SpinType}
            }
            
            for osc in self.oscillators:
                stats['spin_counts'][osc.spin.name] += 1
            
            history.append(stats)
            
            if current_temp < 0.1 and step > 100:
                break
        
        return history
    
    def visualize_evolution(self, history: List[Dict]):
        """Визуализация эволюции спинов."""
        fig = plt.figure(figsize=(15, 10))
        
        steps = [h['step'] for h in history]
        temps = [h['temperature'] for h in history]
        
        # 1. Распределение спинов
        ax1 = plt.subplot(2, 2, 1)
        for spin in SpinType:
            counts = [h['spin_counts'][spin.name] for h in history]
            ax1.plot(steps, counts, label=spin.name, linewidth=2)
        
        ax1.set_xlabel('Шаги эволюции')
        ax1.set_ylabel('Количество осцилляторов')
        ax1.set_title('Эволюция спинов')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Температура
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(steps, temps, 'r-', linewidth=2)
        ax2.set_xlabel('Время')
        ax2.set_ylabel('Температура', color='r')
        ax2.set_title('Остывание Вселенной')
        ax2.grid(True, alpha=0.3)
        
        # Точка фазового перехода
        critical_idx = np.argmin(np.abs(np.array(temps) - 0.5))
        ax2.axvline(x=steps[critical_idx], color='g', linestyle='--', alpha=0.5)
        ax2.text(steps[critical_idx], max(temps)*0.8, 
                'Фазовый\nпереход', ha='center', fontsize=9)
        
        # 3. Энергетическое распределение
        ax3 = plt.subplot(2, 2, 3)
        
        fermion_energies = [
            np.abs(o.amplitude)**2 
            for o in self.oscillators 
            if o.spin == SpinType.SPINOR
        ]
        boson_energies = [
            np.abs(o.amplitude)**2 
            for o in self.oscillators 
            if o.spin in [SpinType.SCALAR, SpinType.VECTOR, SpinType.TENSOR]
        ]
        
        if fermion_energies:
            ax3.hist(fermion_energies, bins=20, alpha=0.7, 
                    label='Фермионы (спин 1/2)', density=True)
        if boson_energies:
            ax3.hist(boson_energies, bins=20, alpha=0.7,
                    label='Бозоны (спин 0,1,2)', density=True)
        
        ax3.set_xlabel('Энергия')
        ax3.set_ylabel('Плотность')
        ax3.set_title('Распределение по энергиям')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Финальное распределение
        ax4 = plt.subplot(2, 2, 4)
        final_counts = history[-1]['spin_counts']
        ax4.bar(final_counts.keys(), final_counts.values(), color=['red', 'blue', 'green', 'purple'])
        ax4.set_xlabel('Тип спина')
        ax4.set_ylabel('Количество')
        ax4.set_title('Финальное распределение')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Статистика
        final_stats = history[-1]
        print("\nФинальная статистика:")
        print(f"Температура: {final_stats['temperature']:.4f}")
        for spin, count in final_stats['spin_counts'].items():
            pct = count / len(self.oscillators) * 100
            print(f"  {spin}: {count} ({pct:.1f}%)")


class DetailedMatterGenesis:
    """
    Полная интегрированная модель рождения материи.
    
    Объединяет все механизмы:
    1. Инфляция и квантовые флуктуации
    2. Параметрический резонанс (разогрев)
    3. Лептогенез и барионная асимметрия
    4. Установление равновесия и нуклеосинтез
    
    Example:
        >>> model = DetailedMatterGenesis()
        >>> results = model.simulate_full_genesis()
    """
    
    def __init__(self):
        """Инициализация всех подсистем."""
        self.resonance_model = ParametricResonance()
        self.leptogenesis_model = LeptogenesisModel()
        self.quantum_model = QuantumCreationInExpandingUniverse()
        
        self.temperature_history = []
        self.particle_species = {}
        self.time = 0
        
    def simulate_full_genesis(self) -> Dict:
        """
        Полная симуляция рождения материи.
        
        Returns:
            словарь со всеми результатами
        """
        print("="*70)
        print("ПОЛНАЯ СИМУЛЯЦИЯ РОЖДЕНИЯ МАТЕРИИ")
        print("="*70)
        
        # 1. Инфляция
        print("\n1. ИНФЛЯЦИОННАЯ ФАЗА:")
        print("   - Квантовые флуктуации метрики")
        print("   - Рождение первичных неоднородностей")
        inflation_results = self._simulate_inflation()
        
        # 2. Разогрев
        print("\n2. ФАЗА РАЗОГРЕВА:")
        print("   - Параметрический резонанс")
        print("   - Распад инфлатона")
        reheating_results = self._simulate_reheating()
        
        # 3. Асимметрия
        print("\n3. ГЕНЕРАЦИЯ АСИММЕТРИИ:")
        print("   - Нарушение CP-симметрии")
        print("   - Лептогенез → Бариогенез")
        asymmetry_results = self._simulate_asymmetry()
        
        # 4. Равновесие
        print("\n4. УСТАНОВЛЕНИЕ РАВНОВЕСИЯ:")
        print("   - Аннигиляция")
        print("   - Нуклеосинтез")
        equilibrium_results = self._simulate_equilibrium()
        
        results = {
            'inflation': inflation_results,
            'reheating': reheating_results,
            'asymmetry': asymmetry_results,
            'equilibrium': equilibrium_results,
            'final_temperature': self.temperature_history[-1] if self.temperature_history else 0,
            'final_baryon_asymmetry': asymmetry_results.get('final_B', 0)
        }
        
        self._visualize_results(results)
        self._print_summary(results)
        
        return results
    
    def _simulate_inflation(self) -> Dict:
        """Симуляция инфляции."""
        N_e_folds = 60
        H_inf = 1e13  # GeV
        
        k_modes = np.logspace(-6, 0, 100)
        P_R = (H_inf**2 / (2*np.pi))**2 * (1 / (2*1e-2))
        spectral_index = 0.96
        power_spectrum = P_R * (k_modes / 0.05)**(spectral_index - 1)
        
        return {
            'N_e_folds': N_e_folds,
            'H_inf': H_inf,
            'power_spectrum': power_spectrum,
            'k_modes': k_modes
        }
    
    def _simulate_reheating(self) -> Dict:
        """Симуляция разогрева."""
        self.resonance_model.simulate_resonance_bands(show_plot=False)
        
        particle_yields = {
            'photons': 1e10,
            'quarks': 1e9,
            'leptons': 1e9,
            'gluons': 1e9,
            'W_Z_bosons': 1e8,
            'higgs': 1e7,
            'dark_matter': 1e8
        }
        
        return {
            'reheating_temperature': 1e9,
            'particle_yields': particle_yields,
            'efficiency': 0.7
        }
    
    def _simulate_asymmetry(self) -> Dict:
        """Симуляция генерации асимметрии."""
        final_B = self.leptogenesis_model.solve_leptogenesis(show_plot=False)
        
        T = 1e12
        alpha_w = 1/30
        sphaleron_rate = 25 * alpha_w**5 * T**4
        
        conversion_efficiency = 0.1
        final_baryon_asymmetry = final_B * conversion_efficiency
        
        return {
            'lepton_asymmetry': final_B,
            'baryon_asymmetry': final_baryon_asymmetry,
            'sphaleron_rate': sphaleron_rate,
            'conversion_efficiency': conversion_efficiency,
            'final_B': final_baryon_asymmetry
        }
    
    def _simulate_equilibrium(self) -> Dict:
        """Симуляция равновесия."""
        # Аннигиляция
        initial_ratio = 1 + 1e-9
        final_ratio = 1 + (initial_ratio - 1) * np.exp(-1e-6 * 1e10)
        
        annihilation = {
            'initial_ratio': initial_ratio,
            'final_ratio': final_ratio,
            'photon_production': 1e9
        }
        
        # CMB
        cmb = {
            'kelvin': 2.725,
            'gev': 2.725 * 8.617e-14,
            'redshift': 1100
        }
        
        # Нуклеосинтез
        nucleosynthesis = {
            'abundances': {
                'H': 0.75,
                'He4': 0.24,
                'D': 0.00003,
                'He3': 0.00001,
                'Li7': 4e-10
            },
            'temperature': 0.1  # MeV
        }
        
        self.temperature_history.append(cmb['gev'])
        
        return {
            'annihilation': annihilation,
            'cmb': cmb,
            'nucleosynthesis': nucleosynthesis
        }
    
    def _visualize_results(self, results: Dict):
        """Визуализация результатов."""
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Временная шкала
        ax1 = plt.subplot(2, 2, 1)
        cosmic_times = [
            (1e-43, 1e19, "Планк"),
            (1e-36, 1e15, "Инфляция"),
            (1e-32, 1e12, "Разогрев"),
            (1e-12, 1e3, "ЭС переход"),
            (1e-6, 0.1, "КХД"),
            (1, 1e-3, "ББН"),
            (1e10, 2.7e-13, "Сейчас")
        ]
        
        times = [t[0] for t in cosmic_times]
        temps = [t[1] for t in cosmic_times]
        labels = [t[2] for t in cosmic_times]
        
        ax1.loglog(times, temps, 'bo-', linewidth=2)
        for t, T, label in cosmic_times:
            ax1.text(t, T, label, fontsize=8, ha='center', va='bottom', rotation=45)
        
        ax1.set_xlabel('Время (сек)')
        ax1.set_ylabel('Температура (GeV)')
        ax1.set_title('Температурная история')
        ax1.grid(True, alpha=0.3)
        
        # 2. Эволюция компонентов
        ax2 = plt.subplot(2, 2, 2)
        
        particle_types = ['Инфлатон', 'Кварки', 'Лептоны', 'Фотоны', 'ТМ']
        stages = ['Инфляция', 'Разогрев', 'Лептогенез', 'Аннигиляция', 'Сейчас']
        
        data = np.array([
            [1.0, 0.1, 0.01, 0.001, 0.0],
            [0.0, 0.8, 0.6, 0.5, 0.4],
            [0.0, 0.7, 0.8, 0.4, 0.3],
            [0.0, 0.5, 0.9, 1.0, 1.0],
            [0.0, 0.2, 0.3, 0.3, 0.27]
        ])
        
        im = ax2.imshow(data, cmap='YlOrRd', aspect='auto')
        ax2.set_xticks(np.arange(len(stages)))
        ax2.set_xticklabels(stages, rotation=45)
        ax2.set_yticks(np.arange(len(particle_types)))
        ax2.set_yticklabels(particle_types)
        ax2.set_title('Эволюция компонентов')
        plt.colorbar(im, ax=ax2, label='Относительная доля')
        
        # 3. Асимметрия
        ax3 = plt.subplot(2, 2, 3)
        
        asym_data = {
            'CP-нарушение': results['asymmetry']['lepton_asymmetry'],
            'Сфалероны': results['asymmetry']['conversion_efficiency'],
            'η (финал)': results['asymmetry']['final_B']
        }
        
        bars = ax3.bar(range(len(asym_data)), list(asym_data.values()))
        ax3.set_xticks(range(len(asym_data)))
        ax3.set_xticklabels(list(asym_data.keys()), rotation=45, ha='right')
        ax3.set_ylabel('Величина')
        ax3.set_title('Генерация асимметрии')
        ax3.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, asym_data.values()):
            if val > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{val:.2e}', ha='center', va='bottom', fontsize=9)
        
        # 4. Современный состав
        ax4 = plt.subplot(2, 2, 4)
        
        composition = {
            'Тёмная энергия': 0.68,
            'Тёмная материя': 0.27,
            'Барионы': 0.05
        }
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        ax4.pie(composition.values(), labels=composition.keys(),
               colors=colors, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Современный состав Вселенной')
        
        plt.tight_layout()
        plt.show()
    
    def _print_summary(self, results: Dict):
        """Вывод сводки."""
        print("\n" + "="*70)
        print("ИТОГОВАЯ СВОДКА")
        print("="*70)
        
        print(f"\n1. ИНФЛЯЦИЯ:")
        print(f"   e-фолды: {results['inflation']['N_e_folds']}")
        print(f"   H: {results['inflation']['H_inf']:.2e} GeV")
        
        print(f"\n2. РАЗОГРЕВ:")
        print(f"   T_reh: {results['reheating']['reheating_temperature']:.2e} GeV")
        print(f"   Эффективность: {results['reheating']['efficiency']*100:.1f}%")
        
        print(f"\n3. АСИММЕТРИЯ:")
        print(f"   L: {results['asymmetry']['lepton_asymmetry']:.2e}")
        print(f"   η: {results['asymmetry']['final_B']:.2e}")
        print(f"   Наблюдаемое: 6×10⁻¹⁰")
        
        observed_B = 6e-10
        simulated_B = results['asymmetry']['final_B']
        
        if simulated_B > 0:
            discrepancy = np.log10(simulated_B / observed_B)
            print(f"\n4. СОГЛАСИЕ:")
            print(f"   Расхождение: {discrepancy:.2f} порядков")
            
            if abs(discrepancy) < 1:
                print("   ✓ Хорошее согласие!")
            else:
                print("   ⚠ Требуется настройка")

