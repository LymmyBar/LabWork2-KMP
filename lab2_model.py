import math
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

G = 9.81  # м/с^2


def deg_to_rad(alpha_deg: float) -> float:
    return math.radians(alpha_deg)


def flight_characteristics(v0: float, alpha_deg: float, h0: float, g: float = G) -> Dict[str, float]:
    alpha = deg_to_rad(alpha_deg)
    v0x = v0 * math.cos(alpha)
    v0y = v0 * math.sin(alpha)

    t_up = v0y / g
    h_max = h0 + (v0 ** 2 * math.sin(alpha) ** 2) / (2 * g)
    disc = v0 ** 2 * math.sin(alpha) ** 2 + 2 * g * h0
    t_total = (v0y + math.sqrt(disc)) / g
    L = v0x * t_total

    return {
        "T": t_total,
        "L": L,
        "Hmax": h_max,
        "t_up": t_up,
        "v0x": v0x,
        "v0y": v0y,
    }


def trajectory_points(v0: float, alpha_deg: float, h0: float, g: float = G, n_points: int = 200):
    params = flight_characteristics(v0, alpha_deg, h0, g)
    T = params["T"]
    alpha = deg_to_rad(alpha_deg)

    t = np.linspace(0, T, n_points)
    x = v0 * math.cos(alpha) * t
    y = h0 + v0 * math.sin(alpha) * t - (g * t ** 2) / 2.0

    return x, y, params


def run_experiments(v0: float, h0_values: List[float], alpha_values: List[float]) -> List[Dict[str, float]]:
    results = []
    for h0 in h0_values:
        for alpha in alpha_values:
            params = flight_characteristics(v0, alpha, h0)
            results.append(
                {
                    "v0": v0,
                    "alpha": alpha,
                    "h0": h0,
                    "T": params["T"],
                    "L": params["L"],
                    "Hmax": params["Hmax"],
                }
            )
    return results


def print_results_table(results: List[Dict[str, float]]):
    table = PrettyTable()
    table.field_names = ["v0, м/с", "α, °", "h0, м", "T, с", "L, м", "Hmax, м"]

    results_sorted = sorted(results, key=lambda r: (r["h0"], r["alpha"]))

    for row in results_sorted:
        table.add_row(
            [
                f"{row['v0']:.1f}",
                f"{row['alpha']:.1f}",
                f"{row['h0']:.1f}",
                f"{row['T']:.3f}",
                f"{row['L']:.3f}",
                f"{row['Hmax']:.3f}",
            ]
        )

    table.float_format = ".3"
    print("\nПідсумкова таблиця результатів обчислювального експерименту:")
    print(table)


def configure_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.margins(x=0, y=0)  # прибираємо відступи від країв осей


def plot_trajectories_for_alphas(v0: float, h0: float, alpha_values: List[float]):
    fig, ax = plt.subplots(figsize=(8, 6))

    for alpha in alpha_values:
        x, y, _ = trajectory_points(v0, alpha, h0)
        ax.plot(x, y, label=f"α = {alpha:.0f}°", linewidth=2)

    configure_axes(ax)
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.set_title(f"Траєкторії руху тіла (v₀ = {v0} м/с, h₀ = {h0} м)")
    ax.legend(title="Кут кидання", loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_L_vs_alpha_for_h0(v0: float, h0_values: List[float], alpha_values: List[float]):
    fig, ax = plt.subplots(figsize=(8, 6))

    for h0 in h0_values:
        L_list = []
        for alpha in alpha_values:
            params = flight_characteristics(v0, alpha, h0)
            L_list.append(params["L"])
        ax.plot(alpha_values, L_list, marker="o", linewidth=2, label=f"h₀ = {h0} м")

    configure_axes(ax)
    ax.set_xlabel("Кут кидання α, °")
    ax.set_ylabel("Дальність польоту L, м")
    ax.set_title(f"Залежність дальності L(α) при v₀ = {v0} м/с для різних h₀")
    ax.legend(title="Початкова висота", loc="best")
    plt.tight_layout()
    plt.show()


def plot_Hmax_vs_alpha_for_h0(v0: float, h0_values: List[float], alpha_values: List[float]):
    fig, ax = plt.subplots(figsize=(8, 6))

    for h0 in h0_values:
        H_list = []
        for alpha in alpha_values:
            params = flight_characteristics(v0, alpha, h0)
            H_list.append(params["Hmax"])
        ax.plot(alpha_values, H_list, marker="s", linewidth=2, label=f"h₀ = {h0} м")

    configure_axes(ax)
    ax.set_xlabel("Кут кидання α, °")
    ax.set_ylabel("Максимальна висота Hmax, м")
    ax.set_title(f"Залежність максимальної висоти Hmax(α) при v₀ = {v0} м/с для різних h₀")
    ax.legend(title="Початкова висота", loc="best")
    plt.tight_layout()
    plt.show()


def check_model_adequacy():
    v0 = 30.0
    alpha = 45.0
    h0 = 10.0

    params = flight_characteristics(v0, alpha, h0)

    print("\nПеревірка адекватності моделі (приклад):")
    print(f"v0 = {v0} м/с, α = {alpha}°, h0 = {h0} м")
    print(f"T      = {params['T']:.5f} с")
    print(f"L      = {params['L']:.5f} м")
    print(f"Hmax   = {params['Hmax']:.5f} м\n")


def main():
    v0 = 40.0
    h0_values = [0.0, 10.0, 20.0]
    alpha_values = list(range(10, 81, 5))

    results = run_experiments(v0=v0, h0_values=h0_values, alpha_values=alpha_values)
    print_results_table(results)

    h0_for_trajectories = 10.0
    alphas_for_trajectories = [20, 30, 40, 50, 60]
    plot_trajectories_for_alphas(v0=v0, h0=h0_for_trajectories, alpha_values=alphas_for_trajectories)

    plot_L_vs_alpha_for_h0(v0=v0, h0_values=h0_values, alpha_values=alpha_values)
    plot_Hmax_vs_alpha_for_h0(v0=v0, h0_values=h0_values, alpha_values=alpha_values)

    check_model_adequacy()


if __name__ == "__main__":
    main()