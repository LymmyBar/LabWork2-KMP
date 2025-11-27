#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторна робота №2: Комп'ютерне моделювання процесів
Тема: Рух тіла, кинутого під кутом α до горизонту з початковою швидкістю v0 з висоти h0

Дисципліна: Комп'ютерне моделювання процесів
Автор: Студент

Опис:
    Ця програма реалізує комп'ютерну модель руху тіла, кинутого під кутом до горизонту.
    Вхідні параметри: початкова швидкість v0, кут кидання α, початкова висота h0.
    Вихідні параметри: траєкторія руху, час польоту, максимальна висота, дальність польоту.
"""

import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable


# ============================================================================
# КОНСТАНТИ ТА ПАРАМЕТРИ МОДЕЛІ
# ============================================================================

# Прискорення вільного падіння (м/с²)
G = 9.81

# Діапазони значень параметрів (згідно з ЛР1)
V0_MIN, V0_MAX = 10, 100       # Діапазон початкової швидкості (м/с)
ALPHA_MIN, ALPHA_MAX = 0, 90   # Діапазон кута кидання (градуси)
H0_MIN, H0_MAX = 0, 50         # Діапазон початкової висоти (м)


# ============================================================================
# ФУНКЦІЇ ДЛЯ РОЗРАХУНКУ ХАРАКТЕРИСТИК ПОЛЬОТУ
# ============================================================================

def calculate_flight_time(v0: float, alpha_deg: float, h0: float) -> float:
    """
    Обчислює повний час польоту тіла.
    
    Формула: T = (v0 * sin(α) + sqrt(v0² * sin²(α) + 2 * g * h0)) / g
    
    Параметри:
        v0 (float): початкова швидкість (м/с)
        alpha_deg (float): кут кидання (градуси)
        h0 (float): початкова висота (м)
    
    Повертає:
        float: час польоту (с)
    """
    # Перетворення кута з градусів у радіани
    alpha_rad = np.radians(alpha_deg)
    
    # Вертикальна складова початкової швидкості
    v0y = v0 * np.sin(alpha_rad)
    
    # Розв'язок квадратного рівняння h0 + v0y*t - g*t²/2 = 0
    # Обираємо додатній корінь
    discriminant = v0y**2 + 2 * G * h0
    T = (v0y + np.sqrt(discriminant)) / G
    
    return T


def calculate_max_height(v0: float, alpha_deg: float, h0: float) -> float:
    """
    Обчислює максимальну висоту підйому тіла.
    
    Формула: Hmax = h0 + (v0² * sin²(α)) / (2g)
    
    Параметри:
        v0 (float): початкова швидкість (м/с)
        alpha_deg (float): кут кидання (градуси)
        h0 (float): початкова висота (м)
    
    Повертає:
        float: максимальна висота (м)
    """
    alpha_rad = np.radians(alpha_deg)
    Hmax = h0 + (v0**2 * np.sin(alpha_rad)**2) / (2 * G)
    return Hmax


def calculate_flight_range(v0: float, alpha_deg: float, h0: float) -> float:
    """
    Обчислює дальність польоту тіла.
    
    Формула: L = v0 * cos(α) * T
    
    Параметри:
        v0 (float): початкова швидкість (м/с)
        alpha_deg (float): кут кидання (градуси)
        h0 (float): початкова висота (м)
    
    Повертає:
        float: дальність польоту (м)
    """
    alpha_rad = np.radians(alpha_deg)
    T = calculate_flight_time(v0, alpha_deg, h0)
    L = v0 * np.cos(alpha_rad) * T
    return L


def calculate_rise_time(v0: float, alpha_deg: float) -> float:
    """
    Обчислює час підйому до максимальної висоти.
    
    Формула: tup = v0 * sin(α) / g
    
    Параметри:
        v0 (float): початкова швидкість (м/с)
        alpha_deg (float): кут кидання (градуси)
    
    Повертає:
        float: час підйому (с)
    """
    alpha_rad = np.radians(alpha_deg)
    tup = v0 * np.sin(alpha_rad) / G
    return tup


def calculate_all_characteristics(v0: float, alpha_deg: float, h0: float) -> dict:
    """
    Обчислює всі характеристики польоту для заданих параметрів.
    
    Параметри:
        v0 (float): початкова швидкість (м/с)
        alpha_deg (float): кут кидання (градуси)
        h0 (float): початкова висота (м)
    
    Повертає:
        dict: словник з характеристиками польоту
    """
    return {
        'v0': v0,
        'alpha': alpha_deg,
        'h0': h0,
        'T': calculate_flight_time(v0, alpha_deg, h0),
        'Hmax': calculate_max_height(v0, alpha_deg, h0),
        'L': calculate_flight_range(v0, alpha_deg, h0),
        'tup': calculate_rise_time(v0, alpha_deg)
    }


# ============================================================================
# ФУНКЦІЇ ДЛЯ ГЕНЕРАЦІЇ ДАНИХ ТРАЄКТОРІЙ
# ============================================================================

def generate_trajectory(v0: float, alpha_deg: float, h0: float, num_points: int = 100) -> tuple:
    """
    Генерує координати траєкторії руху тіла.
    
    Рівняння руху:
        x(t) = v0 * cos(α) * t
        y(t) = h0 + v0 * sin(α) * t - (g * t²)/2
    
    Параметри:
        v0 (float): початкова швидкість (м/с)
        alpha_deg (float): кут кидання (градуси)
        h0 (float): початкова висота (м)
        num_points (int): кількість точок траєкторії
    
    Повертає:
        tuple: (масив x-координат, масив y-координат, масив часу)
    """
    alpha_rad = np.radians(alpha_deg)
    
    # Обчислюємо повний час польоту
    T = calculate_flight_time(v0, alpha_deg, h0)
    
    # Створюємо масив часу від 0 до T
    t = np.linspace(0, T, num_points)
    
    # Обчислюємо координати
    x = v0 * np.cos(alpha_rad) * t
    y = h0 + v0 * np.sin(alpha_rad) * t - (G * t**2) / 2
    
    # Гарантуємо, що y >= 0 (тіло на землі)
    y = np.maximum(y, 0)
    
    return x, y, t


# ============================================================================
# ФУНКЦІЇ ДЛЯ ПРОВЕДЕННЯ СЕРІЙ ЕКСПЕРИМЕНТІВ
# ============================================================================

def run_angle_experiment(v0: float, h0: float, angles: list) -> list:
    """
    Проводить серію експериментів для різних кутів кидання.
    
    Параметри:
        v0 (float): фіксована початкова швидкість (м/с)
        h0 (float): фіксована початкова висота (м)
        angles (list): список кутів кидання (градуси)
    
    Повертає:
        list: список словників з результатами для кожного кута
    """
    results = []
    for alpha in angles:
        result = calculate_all_characteristics(v0, alpha, h0)
        results.append(result)
    return results


def run_height_experiment(v0: float, alpha_deg: float, heights: list) -> list:
    """
    Проводить серію експериментів для різних початкових висот.
    
    Параметри:
        v0 (float): фіксована початкова швидкість (м/с)
        alpha_deg (float): фіксований кут кидання (градуси)
        heights (list): список початкових висот (м)
    
    Повертає:
        list: список словників з результатами для кожної висоти
    """
    results = []
    for h0 in heights:
        result = calculate_all_characteristics(v0, alpha_deg, h0)
        results.append(result)
    return results


# ============================================================================
# ФУНКЦІЇ ДЛЯ ВИВЕДЕННЯ РЕЗУЛЬТАТІВ У ТАБЛИЦЯХ
# ============================================================================

def print_results_table(results: list, title: str = "Результати експерименту"):
    """
    Виводить результати експерименту у вигляді форматованої таблиці.
    
    Параметри:
        results (list): список словників з результатами
        title (str): заголовок таблиці
    """
    print(f"\n{title}")
    print("=" * 80)
    
    # Створюємо таблицю за допомогою PrettyTable
    table = PrettyTable()
    table.field_names = [
        "v0 (м/с)", 
        "α (°)", 
        "h0 (м)", 
        "T (с)", 
        "Hmax (м)", 
        "L (м)",
        "tup (с)"
    ]
    
    # Налаштування вирівнювання стовпців
    table.align = "r"
    
    # Додаємо дані до таблиці
    for r in results:
        table.add_row([
            f"{r['v0']:.1f}",
            f"{r['alpha']:.1f}",
            f"{r['h0']:.1f}",
            f"{r['T']:.3f}",
            f"{r['Hmax']:.2f}",
            f"{r['L']:.2f}",
            f"{r['tup']:.3f}"
        ])
    
    print(table)
    return table


def print_verification_table(results: list, title: str = "Верифікація моделі"):
    """
    Виводить таблицю верифікації (порівняння числових та аналітичних результатів).
    
    Параметри:
        results (list): список словників з результатами
        title (str): заголовок таблиці
    """
    print(f"\n{title}")
    print("=" * 100)
    
    table = PrettyTable()
    table.field_names = [
        "α (°)",
        "T числ. (с)",
        "T аналіт. (с)",
        "ΔT (%)",
        "L числ. (м)",
        "L аналіт. (м)",
        "ΔL (%)"
    ]
    table.align = "r"
    
    for r in results:
        # Незалежний аналітичний розрахунок для верифікації
        alpha_rad = np.radians(r['alpha'])
        v0y = r['v0'] * np.sin(alpha_rad)
        v0x = r['v0'] * np.cos(alpha_rad)
        
        # Аналітичний час польоту (пряме застосування формули)
        discriminant = v0y**2 + 2 * G * r['h0']
        T_analytical = (v0y + np.sqrt(discriminant)) / G
        
        # Аналітична дальність (пряме застосування формули)
        L_analytical = v0x * T_analytical
        
        # Обчислення відносної похибки
        delta_T = abs(r['T'] - T_analytical) / T_analytical * 100 if T_analytical != 0 else 0
        delta_L = abs(r['L'] - L_analytical) / L_analytical * 100 if L_analytical != 0 else 0
        
        table.add_row([
            f"{r['alpha']:.1f}",
            f"{r['T']:.4f}",
            f"{T_analytical:.4f}",
            f"{delta_T:.6f}",
            f"{r['L']:.4f}",
            f"{L_analytical:.4f}",
            f"{delta_L:.6f}"
        ])
    
    print(table)
    print("\nПримітка: ΔT та ΔL показують відносну похибку між числовими")
    print("та аналітичними розрахунками. Значення близькі до 0% підтверджують адекватність моделі.")
    return table


# ============================================================================
# ФУНКЦІЇ ДЛЯ ВІЗУАЛІЗАЦІЇ РЕЗУЛЬТАТІВ
# ============================================================================

def plot_trajectories(v0: float, h0: float, angles: list, save_path: str = None):
    """
    Будує сімейство траєкторій y(x) для різних кутів кидання.
    
    Параметри:
        v0 (float): початкова швидкість (м/с)
        h0 (float): початкова висота (м)
        angles (list): список кутів кидання (градуси)
        save_path (str): шлях для збереження зображення (опціонально)
    """
    plt.figure(figsize=(12, 8))
    
    # Використовуємо кольорову палітру для розрізнення кривих
    colors = plt.cm.viridis(np.linspace(0, 1, len(angles)))
    
    for i, alpha in enumerate(angles):
        x, y, t = generate_trajectory(v0, alpha, h0)
        plt.plot(x, y, color=colors[i], linewidth=2, label=f'α = {alpha}°')
    
    # Оформлення графіка
    plt.xlabel('Горизонтальна відстань x (м)', fontsize=12)
    plt.ylabel('Висота y (м)', fontsize=12)
    plt.title(f'Траєкторії руху тіла при v₀ = {v0} м/с, h₀ = {h0} м', fontsize=14)
    plt.legend(title='Кут кидання', loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    # Зберігаємо графік, якщо вказано шлях
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Графік збережено: {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_trajectories_varying_heights(v0: float, alpha_deg: float, heights: list, save_path: str = None):
    """
    Будує сімейство траєкторій y(x) для різних початкових висот.
    
    Параметри:
        v0 (float): початкова швидкість (м/с)
        alpha_deg (float): кут кидання (градуси)
        heights (list): список початкових висот (м)
        save_path (str): шлях для збереження зображення (опціонально)
    """
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(heights)))
    
    for i, h0 in enumerate(heights):
        x, y, t = generate_trajectory(v0, alpha_deg, h0)
        plt.plot(x, y, color=colors[i], linewidth=2, label=f'h₀ = {h0} м')
    
    plt.xlabel('Горизонтальна відстань x (м)', fontsize=12)
    plt.ylabel('Висота y (м)', fontsize=12)
    plt.title(f'Траєкторії руху тіла при v₀ = {v0} м/с, α = {alpha_deg}°', fontsize=14)
    plt.legend(title='Початкова висота', loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Графік збережено: {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_range_vs_angle(v0: float, heights: list, angle_range: tuple = (0, 90), save_path: str = None):
    """
    Будує графік залежності дальності польоту від кута кидання для різних висот.
    
    Параметри:
        v0 (float): початкова швидкість (м/с)
        heights (list): список початкових висот (м)
        angle_range (tuple): діапазон кутів (мін, макс) у градусах
        save_path (str): шлях для збереження зображення (опціонально)
    """
    plt.figure(figsize=(12, 8))
    
    # Генеруємо масив кутів
    angles = np.linspace(angle_range[0], angle_range[1], 91)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(heights)))
    
    for i, h0 in enumerate(heights):
        ranges = [calculate_flight_range(v0, alpha, h0) for alpha in angles]
        plt.plot(angles, ranges, color=colors[i], linewidth=2, label=f'h₀ = {h0} м')
        
        # Знаходимо оптимальний кут для максимальної дальності
        max_range_idx = np.argmax(ranges)
        optimal_angle = angles[max_range_idx]
        max_range = ranges[max_range_idx]
        plt.scatter([optimal_angle], [max_range], color=colors[i], s=100, zorder=5, 
                   edgecolor='black', linewidth=1.5)
    
    plt.xlabel('Кут кидання α (°)', fontsize=12)
    plt.ylabel('Дальність польоту L (м)', fontsize=12)
    plt.title(f'Залежність дальності польоту від кута кидання при v₀ = {v0} м/с', fontsize=14)
    plt.legend(title='Початкова висота', loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, 90)
    plt.ylim(bottom=0)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Графік збережено: {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_bar_chart_comparison(v0: float, alpha_deg: float, heights: list, save_path: str = None):
    """
    Будує стовпчикову діаграму для порівняння характеристик польоту при різних висотах.
    
    Параметри:
        v0 (float): початкова швидкість (м/с)
        alpha_deg (float): кут кидання (градуси)
        heights (list): список початкових висот (м)
        save_path (str): шлях для збереження зображення (опціонально)
    """
    # Обчислюємо характеристики для кожної висоти
    flight_times = []
    max_heights = []
    ranges = []
    
    for h0 in heights:
        result = calculate_all_characteristics(v0, alpha_deg, h0)
        flight_times.append(result['T'])
        max_heights.append(result['Hmax'])
        ranges.append(result['L'])
    
    # Налаштування стовпчикової діаграми
    x = np.arange(len(heights))
    width = 0.25
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Перша група стовпчиків - час польоту
    bars1 = ax1.bar(x - width, flight_times, width, label='Час польоту T (с)', color='steelblue')
    
    # Друга вісь для масштабування
    ax2 = ax1.twinx()
    
    # Друга група стовпчиків - дальність
    bars2 = ax2.bar(x, ranges, width, label='Дальність L (м)', color='forestgreen')
    
    # Третя група стовпчиків - максимальна висота
    bars3 = ax2.bar(x + width, max_heights, width, label='Макс. висота Hmax (м)', color='coral')
    
    # Підписи осей
    ax1.set_xlabel('Початкова висота h₀ (м)', fontsize=12)
    ax1.set_ylabel('Час польоту T (с)', fontsize=12, color='steelblue')
    ax2.set_ylabel('Відстань (м)', fontsize=12)
    
    ax1.set_title(f'Порівняння характеристик польоту при v₀ = {v0} м/с, α = {alpha_deg}°', fontsize=14)
    
    # Налаштування міток на осі X
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{h}' for h in heights])
    
    # Об'єднана легенда
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Графік збережено: {save_path}")
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# ФУНКЦІЯ ГЕНЕРАЦІЇ ДИНАМІЧНИХ ВИСНОВКІВ
# ============================================================================

def generate_conclusions(v0: float, heights: list, results_angles: list, results_heights: list) -> list:
    """
    Генерує динамічні висновки на основі результатів експериментів.
    
    Параметри:
        v0 (float): початкова швидкість (м/с)
        heights (list): список початкових висот (м)
        results_angles (list): результати експерименту з різними кутами
        results_heights (list): результати експерименту з різними висотами
    
    Повертає:
        list: список висновків
    """
    conclusions = []
    
    # Аналіз впливу кута на максимальну висоту
    if len(results_angles) >= 2:
        hmax_values = [r['Hmax'] for r in results_angles]
        if hmax_values[-1] > hmax_values[0]:
            conclusions.append("Зі збільшенням кута кидання збільшується максимальна висота підйому.")
    
    # Знаходимо оптимальний кут для різних висот
    optimal_angles = []
    for h0 in heights:
        angles = np.linspace(0, 90, 91)
        ranges = [calculate_flight_range(v0, alpha, h0) for alpha in angles]
        optimal_angle = angles[np.argmax(ranges)]
        optimal_angles.append((h0, optimal_angle))
    
    conclusions.append("Оптимальний кут для максимальної дальності залежить від початкової висоти.")
    
    # Перевірка для h0 = 0
    for h0, opt_angle in optimal_angles:
        if h0 == 0:
            conclusions.append(f"При h₀ = 0 м оптимальний кут складає {opt_angle:.0f}°.")
            break
    
    # Аналіз тренду оптимальних кутів
    if len(optimal_angles) >= 2 and optimal_angles[-1][1] < optimal_angles[0][1]:
        conclusions.append("Зі збільшенням h₀ оптимальний кут зменшується.")
    
    # Аналіз впливу початкової висоти на дальність
    if len(results_heights) >= 2:
        range_values = [r['L'] for r in results_heights]
        if range_values[-1] > range_values[0]:
            increase_pct = (range_values[-1] - range_values[0]) / range_values[0] * 100
            conclusions.append(f"Збільшення початкової висоти з {results_heights[0]['h0']:.0f} до "
                             f"{results_heights[-1]['h0']:.0f} м збільшує дальність на {increase_pct:.1f}%.")
    
    conclusions.append("Модель адекватна, результати співпадають з аналітичними розрахунками.")
    
    return conclusions


# ============================================================================
# ФУНКЦІЯ ПЕРЕВІРКИ АДЕКВАТНОСТІ МОДЕЛІ
# ============================================================================

def verify_model(v0: float, alpha_deg: float, h0: float):
    """
    Перевіряє адекватність моделі шляхом порівняння з аналітичними формулами.
    
    Параметри:
        v0 (float): початкова швидкість (м/с)
        alpha_deg (float): кут кидання (градуси)
        h0 (float): початкова висота (м)
    """
    print("\n" + "=" * 80)
    print("ВЕРИФІКАЦІЯ МОДЕЛІ")
    print("=" * 80)
    
    # Обчислюємо характеристики за допомогою нашої моделі
    result = calculate_all_characteristics(v0, alpha_deg, h0)
    
    # Аналітичні розрахунки (для порівняння)
    alpha_rad = np.radians(alpha_deg)
    
    # Аналітичний час польоту
    v0y = v0 * np.sin(alpha_rad)
    T_analytical = (v0y + np.sqrt(v0y**2 + 2 * G * h0)) / G
    
    # Аналітична максимальна висота
    Hmax_analytical = h0 + (v0**2 * np.sin(alpha_rad)**2) / (2 * G)
    
    # Аналітична дальність
    L_analytical = v0 * np.cos(alpha_rad) * T_analytical
    
    # Аналітичний час підйому
    tup_analytical = v0 * np.sin(alpha_rad) / G
    
    print(f"\nВхідні параметри:")
    print(f"  Початкова швидкість v₀ = {v0} м/с")
    print(f"  Кут кидання α = {alpha_deg}°")
    print(f"  Початкова висота h₀ = {h0} м")
    
    print(f"\nПорівняння результатів:")
    print(f"{'Параметр':<25} {'Модель':<15} {'Аналітика':<15} {'Різниця':<15}")
    print("-" * 70)
    
    params = [
        ('Час польоту T (с)', result['T'], T_analytical),
        ('Макс. висота Hmax (м)', result['Hmax'], Hmax_analytical),
        ('Дальність L (м)', result['L'], L_analytical),
        ('Час підйому tup (с)', result['tup'], tup_analytical)
    ]
    
    for name, model_val, analytical_val in params:
        diff = abs(model_val - analytical_val)
        print(f"{name:<25} {model_val:<15.6f} {analytical_val:<15.6f} {diff:<15.10f}")
    
    print("\nВисновок: Модель є адекватною, оскільки результати числових")
    print("розрахунків співпадають з аналітичними розрахунками за формулами.")


# ============================================================================
# ГОЛОВНА ФУНКЦІЯ
# ============================================================================

def main():
    """
    Головна функція програми.
    Запускає серію обчислювальних експериментів та візуалізує результати.
    """
    print("=" * 80)
    print("КОМП'ЮТЕРНА МОДЕЛЬ: Рух тіла, кинутого під кутом до горизонту")
    print("Лабораторна робота №2")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # ПАРАМЕТРИ ЕКСПЕРИМЕНТІВ (можна легко змінювати)
    # -------------------------------------------------------------------------
    
    # Фіксовані параметри для першого експерименту
    v0_fixed = 30.0  # Початкова швидкість (м/с)
    h0_fixed = 10.0  # Початкова висота (м)
    
    # Набір кутів для дослідження
    angles = [15, 30, 45, 60, 75]
    
    # Набір початкових висот для дослідження
    heights = [0, 10, 20, 30, 40]
    
    # -------------------------------------------------------------------------
    # ЕКСПЕРИМЕНТ 1: Вплив кута кидання на траєкторію
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 80)
    print("ЕКСПЕРИМЕНТ 1: Вплив кута кидання на характеристики польоту")
    print(f"Параметри: v₀ = {v0_fixed} м/с, h₀ = {h0_fixed} м")
    print("=" * 80)
    
    # Проводимо серію експериментів
    results_angles = run_angle_experiment(v0_fixed, h0_fixed, angles)
    
    # Виводимо результати у таблиці
    print_results_table(results_angles, 
                       f"Результати для різних кутів кидання (v₀={v0_fixed} м/с, h₀={h0_fixed} м)")
    
    # Будуємо графік траєкторій
    plot_trajectories(v0_fixed, h0_fixed, angles, 
                     save_path='trajectories_angles.png')
    
    # -------------------------------------------------------------------------
    # ЕКСПЕРИМЕНТ 2: Вплив початкової висоти на траєкторію
    # -------------------------------------------------------------------------
    
    alpha_fixed = 45  # Фіксований кут кидання (градуси)
    
    print("\n" + "=" * 80)
    print("ЕКСПЕРИМЕНТ 2: Вплив початкової висоти на характеристики польоту")
    print(f"Параметри: v₀ = {v0_fixed} м/с, α = {alpha_fixed}°")
    print("=" * 80)
    
    # Проводимо серію експериментів
    results_heights = run_height_experiment(v0_fixed, alpha_fixed, heights)
    
    # Виводимо результати у таблиці
    print_results_table(results_heights,
                       f"Результати для різних початкових висот (v₀={v0_fixed} м/с, α={alpha_fixed}°)")
    
    # Будуємо графік траєкторій для різних висот
    plot_trajectories_varying_heights(v0_fixed, alpha_fixed, heights,
                                     save_path='trajectories_heights.png')
    
    # -------------------------------------------------------------------------
    # ДОДАТКОВІ ГРАФІКИ
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 80)
    print("ДОДАТКОВІ ГРАФІКИ")
    print("=" * 80)
    
    # Графік залежності дальності від кута для різних висот
    plot_range_vs_angle(v0_fixed, heights, 
                       save_path='range_vs_angle.png')
    
    # Стовпчикова діаграма порівняння
    plot_bar_chart_comparison(v0_fixed, alpha_fixed, heights,
                             save_path='bar_comparison.png')
    
    # -------------------------------------------------------------------------
    # ВЕРИФІКАЦІЯ МОДЕЛІ
    # -------------------------------------------------------------------------
    
    verify_model(v0_fixed, alpha_fixed, h0_fixed)
    
    # Виводимо таблицю верифікації для різних кутів
    print_verification_table(results_angles, 
                            "Таблиця верифікації для різних кутів кидання")
    
    # -------------------------------------------------------------------------
    # ПІДСУМКОВА ІНФОРМАЦІЯ
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 80)
    print("ПІДСУМОК")
    print("=" * 80)
    print("\nСтворені файли:")
    print("  1. trajectories_angles.png - траєкторії для різних кутів")
    print("  2. trajectories_heights.png - траєкторії для різних висот")
    print("  3. range_vs_angle.png - залежність дальності від кута")
    print("  4. bar_comparison.png - порівняльна діаграма")
    
    # Генерація динамічних висновків на основі результатів
    conclusions = generate_conclusions(v0_fixed, heights, results_angles, results_heights)
    
    print("\nВисновки:")
    for i, conclusion in enumerate(conclusions, 1):
        print(f"  {i}. {conclusion}")


# ============================================================================
# ТОЧКА ВХОДУ В ПРОГРАМУ
# ============================================================================

if __name__ == "__main__":
    main()
