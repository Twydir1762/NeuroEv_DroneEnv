import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import math

def plot_learning_curve(data_dict: dict, window=150,
                        colors=((83,74,183), (15,110,86), (186,69,23)),
                        font_family="Times New Roman"):

    rollings = {key: pd.Series(data_dict[key]).rolling(window=window, min_periods=1).mean() for key in data_dict}
    colors_dict = {key: color for key, color in zip(data_dict.keys(), colors)}

    fig = go.Figure()

    for agent_name, rewards_data in data_dict.items():

        plt_color = colors_dict[agent_name]

        # Линия сырых данных
        fig.add_trace(
            go.Scatter(
                y=rewards_data,
                mode="lines",
                name=f"Награды {agent_name}",
                line=dict(color=f'rgba({plt_color[0]},{plt_color[1]},{plt_color[2]},0.3)'),
                visible="legendonly",
            )
        )

        # Линия сглаженных данных
        fig.add_trace(
            go.Scatter(
                y=rollings[agent_name],
                mode="lines",
                name=f"Награды {agent_name} (Окно {window})",
                line=dict(color=f'rgb({plt_color[0]},{plt_color[1]},{plt_color[2]})', width=2),
            )
        )

    fig.update_xaxes(
        mirror=True, ticks='outside',
        ticklen=12, tickcolor='rgba(0,0,0,0)',
        showline=True, linecolor='black', linewidth=1,
        gridcolor='rgba(150,150,150,0.25)',
        minor=dict(showgrid=True, gridcolor='rgba(150,150,150,0.1)')
    )

    fig.update_yaxes(
        mirror=True, ticks='outside',
        ticklen=12, tickcolor='rgba(0,0,0,0)',
        showline=True, linecolor='black', linewidth=1,
        gridcolor='rgba(150,150,150,0.25)',
        minor=dict(showgrid=True, gridcolor='rgba(150,150,150,0.1)')
    )

    fig.update_layout(
        title=dict(text="Кривая обучения (Learning Curve)", font=dict(size=30)),
        xaxis_title=dict(text="Эпизоды", font=dict(size=22)),
        yaxis_title=dict(text="Суммарная награда", font=dict(size=22)),
        template="plotly_white",
        legend=dict(
            x=1.02, y=1,
            font=dict(
                family=font_family,
                size=14,
                color="black"
            ),
            traceorder="normal",
            bordercolor="black",
            borderwidth=1
        ),
        font_family=font_family,
        font=dict(size=18),

    )

    return fig


def plot_joint_cdf(win_steps_dict: dict,
                   colors=((83, 74, 183), (15, 110, 86), (186, 69, 23)),
                   font_family="Times New Roman"):
    colors_dict = {key: color for key, color in zip(win_steps_dict.keys(), colors)}
    fig = go.Figure()

    for agent_name, steps_list in win_steps_dict.items():
        if not steps_list:  # Если агент ни разу не победил, пропускаем
            print(f"У {agent_name} нет успешных шагов для CDF.")
            continue

        sorted_steps = np.sort(steps_list)
        # Кумулятивная вероятность от 1/N до 1.0
        y_vals = np.arange(1, len(sorted_steps) + 1) / len(sorted_steps)

        x_plot = np.concatenate([[sorted_steps[0] - 1], sorted_steps])
        y_plot = np.concatenate([[0], y_vals])

        plt_color = colors_dict[agent_name]

        # CDF
        fig.add_trace(
            go.Scatter(
                x=x_plot,
                y=y_plot,
                mode="lines",
                name=f"CDF {agent_name}",
                line=dict(color=f'rgb({plt_color[0]},{plt_color[1]},{plt_color[2]})', width=2.5, shape='hv'),
            )
        )

    fig.update_xaxes(
        mirror=True, ticks='outside',
        ticklen=12, tickcolor='rgba(0,0,0,0)',
        showline=True, linecolor='black', linewidth=1,
        gridcolor='rgba(150,150,150,0.25)',
        minor=dict(showgrid=True, gridcolor='rgba(150,150,150,0.1)')
    )

    fig.update_yaxes(
        mirror=True, ticks='outside',
        ticklen=12, tickcolor='rgba(0,0,0,0)',
        showline=True, linecolor='black', linewidth=1,
        gridcolor='rgba(150,150,150,0.25)',
        minor=dict(showgrid=True, gridcolor='rgba(150,150,150,0.1)', dtick=0.05),
        range=[0, 1.05]  # Ограничиваем ось Y от 0 до 1
    )

    fig.update_layout(
        title=dict(text="Функция распределения времени успеха (CDF)", font=dict(size=30)),
        xaxis_title=dict(text="Количество шагов до цели", font=dict(size=22)),
        yaxis_title=dict(text="F(x) = Вероятность успеха", font=dict(size=22)),
        template="plotly_white",
        legend=dict(
            x=1.02, y=1,
            font=dict(family=font_family, size=14, color="black"),
            traceorder="normal", bordercolor="black", borderwidth=1
        ),
        font_family=font_family,
        font=dict(size=18),
    )

    return fig


def plot_charge_heatmap(final_positions_dict: dict, final_charges_dict: dict,
                        map_size=(24, 12), grid_resolution=0.5,
                        obstacles=None, charge_pos=(21.5, 8.5), charge_radius=0.9,
                        font_family="Times New Roman", subplot_width=500, max_cols=1):

    cols = int(map_size[0] / grid_resolution)
    rows = int(map_size[1] / grid_resolution)

    agent_names = list(final_positions_dict.keys())
    n = len(agent_names)

    # Умная сетка (перенос)
    plot_cols = min(n, max_cols)
    plot_rows = math.ceil(n / max_cols)

    fig = make_subplots(rows=plot_rows, cols=plot_cols,
                        subplot_titles=agent_names,
                        horizontal_spacing=0.08,
                        vertical_spacing=0.15)

    # ширина/высота динамически
    map_ratio = map_size[1] / map_size[0]  # Соотношение сторон карты (12/24 = 0.5)
    subplot_height = subplot_width * map_ratio  # Высота одного графика (500 * 0.5 = 250px)

    fig_width = plot_cols * subplot_width + 150
    fig_height = plot_rows * (subplot_height + 100) + 100  # +100px на заголовки и общие отступы

    for idx, agent_name in enumerate(agent_names):
        positions = final_positions_dict[agent_name]
        charges = final_charges_dict[agent_name]

        current_row = (idx // max_cols) + 1
        current_col = (idx % max_cols) + 1

        grid_sum = np.zeros((rows, cols))
        grid_cnt = np.zeros((rows, cols))

        for (x, y), charge in zip(positions, charges):
            col_i = int(min(x / grid_resolution, cols - 1))
            row_i = int(min(y / grid_resolution, rows - 1))
            grid_sum[row_i, col_i] += charge
            grid_cnt[row_i, col_i] += 1

        with np.errstate(invalid='ignore'):
            grid_avg = np.where(grid_cnt > 0, grid_sum / grid_cnt, np.nan)

        fig.add_trace(
            go.Heatmap(
                z=np.flipud(grid_avg),
                x=np.linspace(0, map_size[0], cols),
                y=np.linspace(map_size[1], 0, rows),
                colorscale="RdYlGn",
                zmin=0, zmax=1,
                colorbar=dict(title="Заряд", x=1.02) if idx == n - 1 else dict(showticklabels=False, thickness=0),
                showscale=(idx == n - 1),
            ),
            row=current_row, col=current_col
        )

        # Препятствия
        if obstacles:
            for (x1, y1, x2, y2) in obstacles:
                fig.add_shape(type="rect",
                              x0=x1, y0=y1, x1=x2, y1=y2,
                              line=dict(color="black", width=1.5),
                              fillcolor="rgba(50,50,50,0.6)",
                              row=current_row, col=current_col)

        # Станция зарядки
        fig.add_shape(type="circle",
                      x0=charge_pos[0] - charge_radius, y0=charge_pos[1] - charge_radius,
                      x1=charge_pos[0] + charge_radius, y1=charge_pos[1] + charge_radius,
                      line=dict(color="blue", width=2),
                      fillcolor="rgba(100,150,255,0.3)",
                      row=current_row, col=current_col)

    # Соотношения сторон
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        constrain="domain"
    )

    fig.update_layout(
        title=dict(text="Heatmap финального заряда", font=dict(size=30)),
        template="plotly_white",
        font_family=font_family,
        font=dict(size=16),
        height=fig_height,
        width=fig_width,
    )

    return fig


def plot_paths(paths_dict: dict, map_size=(24, 12),
                      colors=((31, 119, 180), (214, 39, 40), (44, 160, 44)),
                      obstacles=None, charge_pos=(21.5, 8.5), charge_radius=0.9,
                      font_family="Times New Roman"):

    fig = go.Figure()

    # Цвета агентов
    agent_colors = {}
    for agent_name, color in zip(paths_dict.keys(), colors):
        agent_colors[agent_name] = f"rgba({color[0]}, {color[1]}, {color[2]}, 0.6)"

    # Препятствия
    if obstacles:
        for (x1, y1, x2, y2) in obstacles:
            fig.add_shape(type="rect",
                          x0=x1, y0=y1, x1=x2, y1=y2,
                          line=dict(color="black", width=1.5),
                          fillcolor="rgba(50,50,50,0.4)")

    # Зарядка
    fig.add_shape(type="circle",
                  x0=charge_pos[0] - charge_radius, y0=charge_pos[1] - charge_radius,
                  x1=charge_pos[0] + charge_radius, y1=charge_pos[1] + charge_radius,
                  line=dict(color="blue", width=2),
                  fillcolor="rgba(100,150,255,0.2)")

    # Траектории
    for agent_name, paths in paths_dict.items():
        color = agent_colors[agent_name]

        for t_idx, path in enumerate(paths):
            if not path:
                continue
            x_coords = [p[0] for p in path]
            y_coords = [p[1] for p in path]

            # Имя в легенде только для первого пути агента
            # show_in_legend = (t_idx == 0)

            fig.add_trace(
                go.Scatter(
                    x=x_coords, y=y_coords,
                    mode='lines+markers',
                    line=dict(width=2.5, color=color),
                    marker=dict(size=4, color=color),
                    name=f"{agent_name}_{t_idx+1}",
                    # legendgroup=agent_name,
                    # showlegend=show_in_legend
                )
            )

    fig.update_xaxes(
        range=(0, map_size[0]),
        mirror=True, ticks='outside',
        ticklen=12, tickcolor='rgba(0,0,0,0)',
        showline=True, linecolor='black', linewidth=1,
        gridcolor='rgba(150,150,150,0.25)',
        minor=dict(showgrid=True, gridcolor='rgba(150,150,150,0.1)')
    )

    fig.update_yaxes(
        range=(0, map_size[1]),
        scaleanchor="x", scaleratio=1,
        mirror=True, ticks='outside',
        ticklen=12, tickcolor='rgba(0,0,0,0)',
        showline=True, linecolor='black', linewidth=1,
        gridcolor='rgba(150,150,150,0.25)',
        minor=dict(showgrid=True, gridcolor='rgba(150,150,150,0.1)')
    )

    fig.update_layout(
        title=dict(text="Траектории движения дрона", font=dict(size=30)),
        template="plotly_white",
        xaxis_title=dict(text="X", font=dict(size=22)),
        yaxis_title=dict(text="Y", font=dict(size=22)),
        font_family=font_family,
        font=dict(size=20),
        margin=dict(r=200),
        legend=dict(
            x=1.02, y=1,
            xanchor="left",
            yanchor="top",
            font=dict(
                family=font_family,
                size=20,
                color="black"
            ),
            traceorder="normal",
            bordercolor="black",
            borderwidth=0.5
        ),
    )

    return fig


def boxplot(discharges_dict):
    box_fig = go.Figure()
    for con_name, data in discharges_dict.items():
        box_fig.add_trace(go.Box(
            y=data,
            name=con_name,
            boxpoints='outliers',  # Показывать только выбросы
        ))

    box_fig.update_layout(
        title="Распределение расхода энергии по агентам (Test)",
        yaxis_title="Израсходованный заряд (1.0 - финальный)",
        xaxis_title="Агент",
        template="plotly_white"
    )
    return box_fig

