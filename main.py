#!/usr/bin/env python3
"""
Главный скрипт для запуска детальной симуляции рождения материи.

Использование:
    python main.py

Этот скрипт запускает полную симуляцию рождения материи во Вселенной,
включая все фазы: инфляцию, разогрев, лептогенез и установление равновесия.
"""

from oscillators import (
    DetailedMatterGenesis,
    ParametricResonance,
    QuantumCreationInExpandingUniverse,
)


def main():
    """Запуск детальной симуляции рождения материи."""
    print("Запуск детальной симуляции рождения материи...")
    print("Это может занять несколько минут...\n")
    
    # Создаем и запускаем полную модель
    genesis_model = DetailedMatterGenesis()
    results = genesis_model.simulate_full_genesis()
    
    # Дополнительный анализ
    print("\n" + "=" * 70)
    print("ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ")
    print("=" * 70)
    
    # Анализ параметрического резонанса
    print("\nАнализ параметрического резонанса:")
    resonance = ParametricResonance()
    
    # Тестируем разные параметры
    test_params = [
        (1e13, 1e-7),
        (1e12, 1e-6),
        (1e14, 1e-8)
    ]
    
    for mass, coupling in test_params:
        resonance.m = mass
        resonance.g = coupling
        
        # Оцениваем скорость рождения
        k_sample = 1.0
        phi_sample = 1e16
        rate = resonance.particle_production_rate(phi_sample, k_sample)
        
        print(f"  M={mass:.1e} GeV, g={coupling:.1e}: dn/dt = {rate:.2e}")
    
    # Анализ квантового рождения
    print("\nАнализ квантового рождения в расширяющейся Вселенной:")
    quantum_model = QuantumCreationInExpandingUniverse()
    quantum_results = quantum_model.analyze_particle_creation(show_plot=False)
    
    print("\nСимуляция завершена успешно!")
    print("Модель демонстрирует все ключевые механизмы рождения материи:")
    print("1. Параметрический резонанс при разогреве")
    print("2. Нарушение CP-симметрии и лептогенез")
    print("3. Квантовые флуктуации в расширяющейся Вселенной")
    print("4. Установление современного состава материи")
    
    return results


if __name__ == "__main__":
    main()
