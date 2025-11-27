# Лабораторна робота 2: Комп'ютерне моделювання

Репозиторій містить програмну реалізацію моделі польоту тіла, кинутого під кутом до горизонту з початкової висоти.

## Запуск проєкту

```bash
# створення та активація віртуального середовища (рекомендується)
python3 -m venv .venv
. .venv/bin/activate

# встановлення залежностей
pip install numpy matplotlib prettytable

# запуск моделі
python lab2_model.py
```

## Блок-схема алгоритму

Нижче наведено блок-схему основного алгоритму обчислення характеристик польоту:

```mermaid
flowchart TD
    Start([Початок]) --> Input[/Ввід: v0, h0,<br/>alpha_values/]
    Input --> Init["results = []<br/>g = 9.81"]
    Init --> LoopH0{"Для кожного h0<br/>у h0_values"}
    LoopH0 -->|Так| LoopAlpha{"Для кожного alpha<br/>у alpha_values"}
    LoopAlpha -->|Так| Convert["alpha_rad =<br/>alpha * pi / 180"]
    Convert --> CalcV["v0x = v0 * cos(alpha_rad)<br/>v0y = v0 * sin(alpha_rad)"]
    CalcV --> CalcTup["t_up = v0y / g"]
    CalcTup --> CalcHmax["Hmax = h0 +<br/>v0^2 * sin^2(alpha_rad) / (2*g)"]
    CalcHmax --> CalcDisc["disc = v0^2 * sin^2(alpha_rad)<br/>+ 2 * g * h0"]
    CalcDisc --> CalcT["T = (v0y + sqrt(disc)) / g"]
    CalcT --> CalcL["L = v0x * T"]
    CalcL --> Save["Зберегти у results:<br/>(v0, alpha, h0, T, L, Hmax)"]
    Save --> LoopAlpha
    LoopAlpha -->|Ні| LoopH0
    LoopH0 -->|Ні| Output[/Вивід таблиці<br/>результатів/]
    Output --> Plot1["Побудова графіка:<br/>траєкторії для різних alpha"]
    Plot1 --> Plot2["Побудова графіка:<br/>L від alpha для різних h0"]
    Plot2 --> Plot3["Побудова графіка:<br/>Hmax від alpha для різних h0"]
    Plot3 --> Check["Перевірка адекватності:<br/>тестовий приклад"]
    Check --> End([Кінець])
```

GitHub відобразить цю діаграму безпосередньо на сторінці репозиторію.