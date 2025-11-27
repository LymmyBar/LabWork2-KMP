#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторна робота №2: Комп'ютерне моделювання процесів
Тема: Комп'ютерна модель руху тіла, кинутого під кутом α до горизонту
      з початковою швидкістю v0 з висоти h0 в полі тяжіння Землі без опору повітря.

Автор: Студент
Дата: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# ==============================================================================
# КОНСТАНТИ ТА ПАРАМЕТРИ МОДЕЛІ
# ==============================================================================

# Прискорення вільного падіння (м/с^2)
G = 9.81

# Діапазони параметрів дослідження
V0_MIN = 10      # Мінімальна початкова швидкість (м/с)
V0_MAX = 100     # Максимальна початкова швидкість (м/с)
ALPHA_MIN = 0    # Мінімальний кут кидання (градуси)
ALPHA_MAX = 90   # Максимальний кут кидання (градуси)
H0_MIN = 0       # Мінімальна початкова висота (м)
H0_MAX = 50      # Максимальна початкова висота (м)

# Параметри за замовчуванням для експериментів
DEFAULT_V0 = 30          # Початкова швидкість (м/с)
DEFAULT_H0 = 0           # Початкова висота (м)
DEFAULT_ALPHA = 45       # Кут кидання (градуси)
ALPHA_STEP = 1           # Крок зміни кута (градуси)

# Налаштування для візуалізації
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# БАЗОВІ ФІЗИЧНІ ФУНКЦІЇ (АНАЛІТИЧНІ ФОРМУЛИ)
# ==============================================================================

def get_velocity_components(v0: float, alpha_deg: float) -> tuple:
    """
    Обчислення компонент початкової швидкості.
    
    Параметри:
        v0: початкова швидкість (м/с)
        alpha_deg: кут кидання (градуси)
    
    Повертає:
        (v0x, v0y): горизонтальна та вертикальна компоненти швидкості (м/с)
    
    Формули:
        v0x = v0 * cos(α)
        v0y = v0 * sin(α)
    """
    alpha_rad = np.radians(alpha_deg)
    v0x = v0 * np.cos(alpha_rad)
    v0y = v0 * np.sin(alpha_rad)
    return v0x, v0y


def get_position(t: float, v0: float, alpha_deg: float, h0: float = 0) -> tuple:
    """
    Обчислення координат тіла в момент часу t.
    
    Параметри:
        t: час (с)
        v0: початкова швидкість (м/с)
        alpha_deg: кут кидання (градуси)
        h0: початкова висота (м)
    
    Повертає:
        (x, y): координати тіла (м)
    
    Формули:
        x(t) = v0 * cos(α) * t
        y(t) = h0 + v0 * sin(α) * t - (g * t^2) / 2
    """
    alpha_rad = np.radians(alpha_deg)
    x = v0 * np.cos(alpha_rad) * t
    y = h0 + v0 * np.sin(alpha_rad) * t - (G * t**2) / 2
    return x, y


def get_time_to_max_height(v0: float, alpha_deg: float) -> float:
    """
    Обчислення часу підйому до максимальної висоти.
    
    Параметри:
        v0: початкова швидкість (м/с)
        alpha_deg: кут кидання (градуси)
    
    Повертає:
        t_up: час підйому (с)
    
    Формула:
        t_up = v0 * sin(α) / g
    """
    alpha_rad = np.radians(alpha_deg)
    t_up = v0 * np.sin(alpha_rad) / G
    return t_up


def get_max_height(v0: float, alpha_deg: float, h0: float = 0) -> float:
    """
    Обчислення максимальної висоти польоту.
    
    Параметри:
        v0: початкова швидкість (м/с)
        alpha_deg: кут кидання (градуси)
        h0: початкова висота (м)
    
    Повертає:
        H_max: максимальна висота (м)
    
    Формула:
        H_max = h0 + v0^2 * sin^2(α) / (2g)
    """
    alpha_rad = np.radians(alpha_deg)
    H_max = h0 + (v0**2 * np.sin(alpha_rad)**2) / (2 * G)
    return H_max


def get_flight_time(v0: float, alpha_deg: float, h0: float = 0) -> float:
    """
    Обчислення повного часу польоту.
    
    Параметри:
        v0: початкова швидкість (м/с)
        alpha_deg: кут кидання (градуси)
        h0: початкова висота (м)
    
    Повертає:
        T: час польоту (с)
    
    Формула:
        T = (v0 * sin(α) + sqrt(v0^2 * sin^2(α) + 2*g*h0)) / g
    """
    alpha_rad = np.radians(alpha_deg)
    sin_alpha = np.sin(alpha_rad)
    discriminant = v0**2 * sin_alpha**2 + 2 * G * h0
    T = (v0 * sin_alpha + np.sqrt(discriminant)) / G
    return T


def get_range(v0: float, alpha_deg: float, h0: float = 0) -> float:
    """
    Обчислення дальності польоту.
    
    Параметри:
        v0: початкова швидкість (м/с)
        alpha_deg: кут кидання (градуси)
        h0: початкова висота (м)
    
    Повертає:
        L: дальність польоту (м)
    
    Формула:
        L = v0 * cos(α) * T
    """
    T = get_flight_time(v0, alpha_deg, h0)
    alpha_rad = np.radians(alpha_deg)
    L = v0 * np.cos(alpha_rad) * T
    return L


def get_trajectory(v0: float, alpha_deg: float, h0: float = 0, num_points: int = 100) -> tuple:
    """
    Обчислення траєкторії польоту.
    
    Параметри:
        v0: початкова швидкість (м/с)
        alpha_deg: кут кидання (градуси)
        h0: початкова висота (м)
        num_points: кількість точок траєкторії
    
    Повертає:
        (x_arr, y_arr): масиви координат траєкторії (м)
    """
    T = get_flight_time(v0, alpha_deg, h0)
    t_arr = np.linspace(0, T, num_points)
    
    alpha_rad = np.radians(alpha_deg)
    x_arr = v0 * np.cos(alpha_rad) * t_arr
    y_arr = h0 + v0 * np.sin(alpha_rad) * t_arr - (G * t_arr**2) / 2
    
    # Забезпечуємо, щоб траєкторія закінчувалася на y = 0
    y_arr = np.maximum(y_arr, 0)
    
    return x_arr, y_arr


# ==============================================================================
# ФУНКЦІЇ ДОСЛІДЖЕННЯ ПАРАМЕТРІВ
# ==============================================================================

def find_optimal_angle(v0: float, h0: float = 0, 
                       alpha_min: float = 0, alpha_max: float = 90, 
                       step: float = 1) -> tuple:
    """
    Знаходження оптимального кута для максимальної дальності польоту.
    
    Параметри:
        v0: початкова швидкість (м/с)
        h0: початкова висота (м)
        alpha_min: мінімальний кут пошуку (градуси)
        alpha_max: максимальний кут пошуку (градуси)
        step: крок пошуку (градуси)
    
    Повертає:
        (optimal_alpha, max_range, flight_time, max_height):
            оптимальний кут (градуси), максимальна дальність (м),
            час польоту (с), максимальна висота (м)
    """
    angles = np.arange(alpha_min, alpha_max + step, step)
    max_range = 0
    optimal_alpha = 0
    
    for alpha in angles:
        L = get_range(v0, alpha, h0)
        if L > max_range:
            max_range = L
            optimal_alpha = alpha
    
    T = get_flight_time(v0, optimal_alpha, h0)
    H_max = get_max_height(v0, optimal_alpha, h0)
    
    return optimal_alpha, max_range, T, H_max


def study_range_vs_angle(v0: float, h0: float = 0, 
                         alpha_min: float = 0, alpha_max: float = 90, 
                         step: float = 1) -> tuple:
    """
    Дослідження залежності дальності польоту від кута кидання.
    
    Параметри:
        v0: початкова швидкість (м/с)
        h0: початкова висота (м)
        alpha_min: мінімальний кут (градуси)
        alpha_max: максимальний кут (градуси)
        step: крок зміни кута (градуси)
    
    Повертає:
        (angles, ranges, times, heights): масиви значень
    """
    angles = np.arange(alpha_min, alpha_max + step, step)
    ranges = np.array([get_range(v0, a, h0) for a in angles])
    times = np.array([get_flight_time(v0, a, h0) for a in angles])
    heights = np.array([get_max_height(v0, a, h0) for a in angles])
    
    return angles, ranges, times, heights


# ==============================================================================
# ФУНКЦІЇ ВІЗУАЛІЗАЦІЇ
# ==============================================================================

def plot_trajectories_by_angle(v0: float, h0: float = 0, 
                                angles: list = None, 
                                save_path: str = None):
    """
    Побудова графіка траєкторій для різних кутів кидання.
    
    Параметри:
        v0: початкова швидкість (м/с)
        h0: початкова висота (м)
        angles: список кутів для відображення (градуси)
        save_path: шлях для збереження графіка
    """
    if angles is None:
        angles = [15, 30, 45, 60, 75]
    
    plt.figure(figsize=(12, 8))
    
    for alpha in angles:
        x, y = get_trajectory(v0, alpha, h0)
        L = get_range(v0, alpha, h0)
        plt.plot(x, y, linewidth=2, label=f'α = {alpha}° (L = {L:.1f} м)')
    
    plt.xlabel('Горизонтальна відстань x, м', fontsize=14)
    plt.ylabel('Висота y, м', fontsize=14)
    plt.title(f'Траєкторії польоту тіла\n(v₀ = {v0} м/с, h₀ = {h0} м)', fontsize=16)
    plt.legend(title='Кут кидання', fontsize=11, title_fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Графік збережено: {save_path}")
    
    plt.show()


def plot_trajectories_by_height(v0: float, alpha_deg: float, 
                                 heights: list = None, 
                                 save_path: str = None):
    """
    Побудова графіка траєкторій для різних початкових висот.
    
    Параметри:
        v0: початкова швидкість (м/с)
        alpha_deg: кут кидання (градуси)
        heights: список початкових висот (м)
        save_path: шлях для збереження графіка
    """
    if heights is None:
        heights = [0, 10, 20, 30, 50]
    
    plt.figure(figsize=(12, 8))
    
    for h0 in heights:
        x, y = get_trajectory(v0, alpha_deg, h0)
        L = get_range(v0, alpha_deg, h0)
        plt.plot(x, y, linewidth=2, label=f'h₀ = {h0} м (L = {L:.1f} м)')
    
    plt.xlabel('Горизонтальна відстань x, м', fontsize=14)
    plt.ylabel('Висота y, м', fontsize=14)
    plt.title(f'Траєкторії польоту тіла\n(v₀ = {v0} м/с, α = {alpha_deg}°)', fontsize=16)
    plt.legend(title='Початкова висота', fontsize=11, title_fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Графік збережено: {save_path}")
    
    plt.show()


def plot_range_vs_angle(v0: float, heights: list = None, 
                        save_path: str = None):
    """
    Побудова графіка залежності дальності від кута для різних висот.
    
    Параметри:
        v0: початкова швидкість (м/с)
        heights: список початкових висот (м)
        save_path: шлях для збереження графіка
    """
    if heights is None:
        heights = [0, 10, 20, 30, 50]
    
    plt.figure(figsize=(12, 8))
    
    for h0 in heights:
        angles, ranges, _, _ = study_range_vs_angle(v0, h0)
        
        # Знаходимо оптимальний кут
        opt_alpha, max_L, _, _ = find_optimal_angle(v0, h0)
        
        plt.plot(angles, ranges, linewidth=2, 
                 label=f'h₀ = {h0} м (αопт = {opt_alpha:.1f}°)')
        
        # Позначаємо максимум
        plt.plot(opt_alpha, max_L, 'o', markersize=8)
    
    plt.xlabel('Кут кидання α, градуси', fontsize=14)
    plt.ylabel('Дальність польоту L, м', fontsize=14)
    plt.title(f'Залежність дальності польоту від кута кидання\n(v₀ = {v0} м/с)', fontsize=16)
    plt.legend(title='Початкова висота', fontsize=11, title_fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, 90)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Графік збережено: {save_path}")
    
    plt.show()


def plot_time_and_height_vs_angle(v0: float, h0: float = 0, 
                                   save_path: str = None):
    """
    Побудова графіків залежності часу польоту та максимальної висоти від кута.
    
    Параметри:
        v0: початкова швидкість (м/с)
        h0: початкова висота (м)
        save_path: шлях для збереження графіка
    """
    angles, ranges, times, heights = study_range_vs_angle(v0, h0)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    
    # Графік дальності
    ax1.plot(angles, ranges, 'b-', linewidth=2)
    ax1.set_xlabel('Кут кидання α, градуси', fontsize=12)
    ax1.set_ylabel('Дальність L, м', fontsize=12)
    ax1.set_title('Дальність польоту L(α)', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xlim(0, 90)
    
    # Графік часу
    ax2.plot(angles, times, 'g-', linewidth=2)
    ax2.set_xlabel('Кут кидання α, градуси', fontsize=12)
    ax2.set_ylabel('Час польоту T, с', fontsize=12)
    ax2.set_title('Час польоту T(α)', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlim(0, 90)
    
    # Графік висоти
    ax3.plot(angles, heights, 'r-', linewidth=2)
    ax3.set_xlabel('Кут кидання α, градуси', fontsize=12)
    ax3.set_ylabel('Максимальна висота Hmax, м', fontsize=12)
    ax3.set_title('Максимальна висота Hmax(α)', fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.set_xlim(0, 90)
    
    plt.suptitle(f'Характеристики польоту тіла (v₀ = {v0} м/с, h₀ = {h0} м)', 
                 fontsize=16, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Графік збережено: {save_path}")
    
    plt.show()


def plot_optimal_angle_bar_chart(v0: float, heights: list = None, 
                                  save_path: str = None):
    """
    Побудова стовпчикової діаграми оптимальних кутів для різних висот.
    
    Параметри:
        v0: початкова швидкість (м/с)
        heights: список початкових висот (м)
        save_path: шлях для збереження графіка
    """
    if heights is None:
        heights = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    
    optimal_angles = []
    max_ranges = []
    
    for h0 in heights:
        opt_alpha, max_L, _, _ = find_optimal_angle(v0, h0)
        optimal_angles.append(opt_alpha)
        max_ranges.append(max_L)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Діаграма оптимальних кутів
    bars1 = ax1.bar(range(len(heights)), optimal_angles, color='steelblue', edgecolor='navy')
    ax1.axhline(y=45, color='red', linestyle='--', linewidth=2, label='α = 45°')
    ax1.set_xticks(range(len(heights)))
    ax1.set_xticklabels([f'{h}' for h in heights])
    ax1.set_xlabel('Початкова висота h₀, м', fontsize=12)
    ax1.set_ylabel('Оптимальний кут αопт, градуси', fontsize=12)
    ax1.set_title('Оптимальний кут кидання', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Діаграма максимальних дальностей
    bars2 = ax2.bar(range(len(heights)), max_ranges, color='forestgreen', edgecolor='darkgreen')
    ax2.set_xticks(range(len(heights)))
    ax2.set_xticklabels([f'{h}' for h in heights])
    ax2.set_xlabel('Початкова висота h₀, м', fontsize=12)
    ax2.set_ylabel('Максимальна дальність Lmax, м', fontsize=12)
    ax2.set_title('Максимальна дальність при оптимальному куті', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.suptitle(f'Залежність оптимальних параметрів від початкової висоти\n(v₀ = {v0} м/с)', 
                 fontsize=16)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Графік збережено: {save_path}")
    
    plt.show()


# ==============================================================================
# ФУНКЦІЇ ВИВЕДЕННЯ РЕЗУЛЬТАТІВ (PrettyTable)
# ==============================================================================

def print_summary_table(v0: float, heights: list = None):
    """
    Виведення підсумкової таблиці результатів з використанням PrettyTable.
    
    Параметри:
        v0: початкова швидкість (м/с)
        heights: список початкових висот для дослідження (м)
    """
    if heights is None:
        heights = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    
    table = PrettyTable()
    table.field_names = ['h₀, м', 'αопт, °', 'Lmax, м', 'T, с', 'Hmax, м', 'Δα від 45°']
    
    for h0 in heights:
        opt_alpha, max_L, T, H_max = find_optimal_angle(v0, h0)
        delta_alpha = opt_alpha - 45
        
        table.add_row([
            f'{h0:.1f}',
            f'{opt_alpha:.2f}',
            f'{max_L:.2f}',
            f'{T:.2f}',
            f'{H_max:.2f}',
            f'{delta_alpha:+.2f}'
        ])
    
    print(f"\n{'='*70}")
    print(f"ПІДСУМКОВІ РЕЗУЛЬТАТИ ДОСЛІДЖЕННЯ")
    print(f"Початкова швидкість v₀ = {v0} м/с, g = {G} м/с²")
    print(f"{'='*70}")
    print(table)
    print(f"{'='*70}\n")
    
    return table


def print_angle_study_table(v0: float, h0: float = 0, step: float = 5):
    """
    Виведення таблиці залежності параметрів польоту від кута.
    
    Параметри:
        v0: початкова швидкість (м/с)
        h0: початкова висота (м)
        step: крок зміни кута (градуси)
    """
    angles, ranges, times, heights = study_range_vs_angle(v0, h0, step=step)
    
    table = PrettyTable()
    table.field_names = ['α, °', 'L, м', 'T, с', 'Hmax, м']
    
    for i in range(len(angles)):
        table.add_row([
            f'{angles[i]:.1f}',
            f'{ranges[i]:.2f}',
            f'{times[i]:.2f}',
            f'{heights[i]:.2f}'
        ])
    
    print(f"\n{'='*50}")
    print(f"ЗАЛЕЖНІСТЬ ПАРАМЕТРІВ ПОЛЬОТУ ВІД КУТА КИДАННЯ")
    print(f"v₀ = {v0} м/с, h₀ = {h0} м, g = {G} м/с²")
    print(f"{'='*50}")
    print(table)
    print(f"{'='*50}\n")
    
    return table


# ==============================================================================
# ФУНКЦІЇ ВЕРИФІКАЦІЇ МОДЕЛІ
# ==============================================================================

def verify_model_adequacy(v0: float = DEFAULT_V0):
    """
    Перевірка адекватності моделі.
    
    Для h0 = 0 теоретично оптимальний кут має бути 45°.
    Для h0 > 0 оптимальний кут має бути менше 45°.
    
    Параметри:
        v0: початкова швидкість для тестування (м/с)
    """
    print("\n" + "="*70)
    print("ПЕРЕВІРКА АДЕКВАТНОСТІ МОДЕЛІ")
    print("="*70)
    
    # Тест 1: При h0 = 0 оптимальний кут ≈ 45°
    print("\n--- Тест 1: При h₀ = 0 оптимальний кут має бути ≈ 45° ---")
    opt_alpha, max_L, T, H_max = find_optimal_angle(v0, h0=0, step=0.1)
    
    print(f"v₀ = {v0} м/с, h₀ = 0 м")
    print(f"Знайдений оптимальний кут: α = {opt_alpha:.2f}°")
    print(f"Теоретичне значення: α = 45°")
    print(f"Відхилення: {abs(opt_alpha - 45):.2f}°")
    
    if abs(opt_alpha - 45) < 1:
        print("✓ ТЕСТ ПРОЙДЕНО: оптимальний кут близький до 45°")
    else:
        print("✗ ТЕСТ НЕ ПРОЙДЕНО: оптимальний кут відрізняється від 45°")
    
    # Тест 2: При h0 > 0 оптимальний кут < 45°
    print("\n--- Тест 2: При h₀ > 0 оптимальний кут має бути < 45° ---")
    
    test_heights = [0, 10, 20, 30, 40, 50]
    previous_angle = 45
    all_decreasing = True
    
    table = PrettyTable()
    table.field_names = ['h₀, м', 'αопт, °', 'Менше 45°?', 'Зменшується?']
    
    for h0 in test_heights:
        opt_alpha, _, _, _ = find_optimal_angle(v0, h0, step=0.1)
        less_than_45 = "Так" if opt_alpha < 45 or h0 == 0 else "Ні"
        decreasing = "Так" if opt_alpha <= previous_angle else "Ні"
        
        if h0 > 0 and opt_alpha >= 45:
            all_decreasing = False
        if opt_alpha > previous_angle and h0 > 0:
            all_decreasing = False
            
        table.add_row([h0, f'{opt_alpha:.2f}', less_than_45, decreasing])
        previous_angle = opt_alpha
    
    print(table)
    
    if all_decreasing:
        print("✓ ТЕСТ ПРОЙДЕНО: оптимальний кут зменшується зі збільшенням h₀")
    else:
        print("✗ ТЕСТ НЕ ПРОЙДЕНО: залежність не підтверджена")
    
    # Тест 3: Порівняння з теоретичними формулами для окремого випадку
    print("\n--- Тест 3: Порівняння з теоретичними формулами ---")
    alpha_test = 30
    h0_test = 0
    
    # Теоретичні значення для h0 = 0
    alpha_rad = np.radians(alpha_test)
    L_theory = (v0**2 * np.sin(2 * alpha_rad)) / G
    T_theory = (2 * v0 * np.sin(alpha_rad)) / G
    H_theory = (v0**2 * np.sin(alpha_rad)**2) / (2 * G)
    
    # Значення з моделі
    L_model = get_range(v0, alpha_test, h0_test)
    T_model = get_flight_time(v0, alpha_test, h0_test)
    H_model = get_max_height(v0, alpha_test, h0_test)
    
    print(f"v₀ = {v0} м/с, α = {alpha_test}°, h₀ = {h0_test} м")
    print(f"\nДальність L:")
    print(f"  Теоретична: {L_theory:.4f} м")
    print(f"  Модель:     {L_model:.4f} м")
    print(f"  Похибка:    {abs(L_theory - L_model):.6f} м")
    
    print(f"\nЧас польоту T:")
    print(f"  Теоретичний: {T_theory:.4f} с")
    print(f"  Модель:      {T_model:.4f} с")
    print(f"  Похибка:     {abs(T_theory - T_model):.6f} с")
    
    print(f"\nМаксимальна висота Hmax:")
    print(f"  Теоретична: {H_theory:.4f} м")
    print(f"  Модель:     {H_model:.4f} м")
    print(f"  Похибка:    {abs(H_theory - H_model):.6f} м")
    
    if (abs(L_theory - L_model) < 0.001 and 
        abs(T_theory - T_model) < 0.001 and 
        abs(H_theory - H_model) < 0.001):
        print("\n✓ ТЕСТ ПРОЙДЕНО: значення моделі збігаються з теоретичними")
    else:
        print("\n✗ ТЕСТ НЕ ПРОЙДЕНО: є розбіжності між моделлю та теорією")
    
    print("\n" + "="*70)


# ==============================================================================
# ОСНОВНА ПРОГРАМА
# ==============================================================================

def run_experiments():
    """
    Проведення основних серій комп'ютерних експериментів.
    """
    print("\n" + "="*70)
    print("ЛАБОРАТОРНА РОБОТА №2")
    print("Комп'ютерне моделювання руху тіла, кинутого під кутом до горизонту")
    print("="*70)
    
    # Параметри експерименту
    v0 = DEFAULT_V0  # 30 м/с
    
    # 1. Виведення таблиці для одного набору параметрів
    print("\n>>> ЕКСПЕРИМЕНТ 1: Залежність параметрів від кута кидання")
    print_angle_study_table(v0, h0=0, step=10)
    
    # 2. Виведення підсумкової таблиці для різних висот
    print("\n>>> ЕКСПЕРИМЕНТ 2: Оптимальний кут для різних початкових висот")
    print_summary_table(v0, heights=[0, 10, 20, 30, 40, 50])
    
    # 3. Побудова графіків траєкторій для різних кутів
    print("\n>>> ЕКСПЕРИМЕНТ 3: Графіки траєкторій для різних кутів")
    plot_trajectories_by_angle(v0, h0=0, angles=[15, 30, 45, 60, 75],
                                save_path='trajectories_by_angle.png')
    
    # 4. Побудова графіків траєкторій для різних висот
    print("\n>>> ЕКСПЕРИМЕНТ 4: Графіки траєкторій для різних початкових висот")
    plot_trajectories_by_height(v0, alpha_deg=45, heights=[0, 10, 20, 30, 50],
                                 save_path='trajectories_by_height.png')
    
    # 5. Побудова графіка L(α) для різних h0
    print("\n>>> ЕКСПЕРИМЕНТ 5: Залежність дальності від кута для різних висот")
    plot_range_vs_angle(v0, heights=[0, 10, 20, 30, 50],
                        save_path='range_vs_angle.png')
    
    # 6. Побудова графіків T(α), Hmax(α), L(α)
    print("\n>>> ЕКСПЕРИМЕНТ 6: Характеристики польоту в залежності від кута")
    plot_time_and_height_vs_angle(v0, h0=10,
                                   save_path='flight_characteristics.png')
    
    # 7. Стовпчикова діаграма оптимальних кутів
    print("\n>>> ЕКСПЕРИМЕНТ 7: Діаграма оптимальних кутів")
    plot_optimal_angle_bar_chart(v0, heights=[0, 10, 20, 30, 40, 50],
                                  save_path='optimal_angle_chart.png')
    
    # 8. Перевірка адекватності моделі
    print("\n>>> ВЕРИФІКАЦІЯ МОДЕЛІ")
    verify_model_adequacy(v0)
    
    print("\n" + "="*70)
    print("ЕКСПЕРИМЕНТИ ЗАВЕРШЕНО")
    print("="*70)


if __name__ == "__main__":
    run_experiments()
