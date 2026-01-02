#!/usr/bin/env python3
"""
Главный скрипт запуска симуляций.

Использование:
    python run_examples.py                    # Запуск основной симуляции
    python run_examples.py --all              # Запуск всех примеров
    python run_examples.py --example NAME     # Запуск конкретного примера
    python run_examples.py --list             # Список примеров
    python run_examples.py --quick            # Быстрая демонстрация

Доступные примеры:
    matter_genesis       - Рождение материи из инфлатона
    spin_dynamics        - Эволюция спинов
    detailed_genesis     - Полная модель genesis
    parametric_resonance - Параметрический резонанс
    leptogenesis         - Генерация асимметрии
    quantum_creation     - Квантовое рождение частиц
"""

import sys
import argparse


def quick_demo():
    """Быстрая демонстрация возможностей библиотеки."""
    from oscillators import (
        MatterGenesisSimulation,
        ParametricResonance,
        PhysicalConstants,
        info
    )
    
    # Показать информацию о библиотеке
    info()
    
    print("\n" + "="*60)
    print("БЫСТРАЯ ДЕМОНСТРАЦИЯ")
    print("="*60)
    
    # 1. Физические константы
    print("\n1. Физические константы:")
    print(f"   Планковская масса: {PhysicalConstants.PLANCK_MASS:.2e} GeV")
    print(f"   Масса Хиггса: {PhysicalConstants.HIGGS_MASS} GeV")
    print(f"   Температура CMB: {PhysicalConstants.CMB_TEMPERATURE} K")
    print(f"   η_B (наблюдаемое): {PhysicalConstants.BARYON_TO_PHOTON_RATIO:.1e}")
    
    # 2. Параметрический резонанс
    print("\n2. Параметрический резонанс:")
    resonance = ParametricResonance(inflaton_mass=1e13, coupling=1e-7)
    rate = resonance.particle_production_rate(phi_amplitude=1e16, k=1.0)
    print(f"   Скорость рождения (k=1): {rate:.2e}")
    
    # 3. Мини-симуляция
    print("\n3. Мини-симуляция (100 шагов):")
    sim = MatterGenesisSimulation(
        volume_size=5.0,
        initial_inflaton_energy=1e10,
        hubble_parameter=1e-4
    )
    
    history = sim.evolve_universe(total_time=100.0, dt=0.5, show_progress=False)
    
    final = history[-1]
    print(f"   Создано частиц: {final['n_particles']}")
    print(f"   Барионная асимметрия: {final['baryon_asymmetry']:.2e}")
    print(f"   Финальная температура: {final['temperature']:.2e} GeV")
    
    print("\n" + "="*60)
    print("Для полной симуляции используйте:")
    print("  python run_examples.py --example matter_genesis")
    print("="*60)


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(
        description="Cosmology - симуляция рождения материи во Вселенной",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--example", "-e",
        type=str,
        help="Название примера для запуска"
    )
    
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Запустить все примеры"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="Показать список примеров"
    )
    
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Быстрая демонстрация"
    )
    
    parser.add_argument(
        "--info", "-i",
        action="store_true",
        help="Информация о библиотеке"
    )
    
    args = parser.parse_args()
    
    # Информация о библиотеке
    if args.info:
        from oscillators import info
        info()
        return
    
    # Список примеров
    if args.list:
        from oscillators.examples import main as examples_main
        sys.argv = ['examples', '--list']
        examples_main()
        return
    
    # Быстрая демонстрация
    if args.quick:
        quick_demo()
        return
    
    # Запуск всех примеров
    if args.all:
        from oscillators.examples import run_all_examples
        run_all_examples()
        return
    
    # Запуск конкретного примера
    if args.example:
        from oscillators.examples import main as examples_main
        sys.argv = ['examples', '--example', args.example]
        examples_main()
        return
    
    # По умолчанию - основная симуляция
    print("\n" + "="*70)
    print("COSMOLOGY - Симуляция рождения материи во Вселенной")
    print("="*70)
    print("\nИспользуйте --help для просмотра доступных опций")
    print("Или --quick для быстрой демонстрации\n")
    
    from oscillators.examples import example_matter_genesis
    example_matter_genesis()


if __name__ == "__main__":
    main()
