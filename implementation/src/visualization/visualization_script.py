from typing import List, Set

import plotly.express as px
import pandas as pd

from cases.case import Case
from solution.solution import Solution
from utils.constants import Constants


def visualize_solution(case: Case, solution: Solution, correct_changepoints: Set[int], real_changepoints_not_found: Set[int]) -> None:
    """
    Visualizes a single case and its solution.
    :param case: The input signal.
    :param solution: A solution with the changepoints found.
    :param correct_changepoints: Changepoints that were correctly found.
    :param real_changepoints_not_found: Changepoints that were not found
    :return: None.
    """
    is_random_signal = case.metadata[0].date == Constants.no_date
    label_x = 'index' if is_random_signal else 'date'
    label_y = 'value' if is_random_signal else 'BPM'
    df_rows = [[metadata.field_from_label(label_x), value] for value, metadata in zip(case.signal, case.metadata)]
    df = pd.DataFrame(df_rows, columns=[label_x, label_y])
    fig = px.line(df, x=label_x, y=label_y, title='Algorithm: ' + solution.metrics.solver_used + ' - Case: ' + case.name)
    for changepoint in solution.changepoints:
        fig.add_vline(case.metadata[changepoint].field_from_label(label_x), line_width=2,
                      line_color='magenta' if changepoint in correct_changepoints else 'red')
    for changepoint in real_changepoints_not_found:
        fig.add_vline(case.metadata[changepoint].field_from_label(label_x), line_width=1,
                      line_color='black')
    fig.show()


def visualize_elbow(case: Case, solution: Solution, guessed_changepoints: int) -> None:
    """
    Visualizes a single case solution objective function for solutions
     with different amount of changepoints.
    :param case: The input signal.
    :param solution: A solution with the metrics for each amount of changepoints.
    :return: None.
    """
    amount_changepoints = len(solution.changepoints)
    df_rows = [[k + 1, solution.metrics.best_prefix[k][case.size]] for k in range(amount_changepoints)]
    df = pd.DataFrame(df_rows, columns=['changepoints', 'objective value'])
    fig = px.line(df, x='changepoints', y='objective value', title='Elbow - ' + solution.metrics.solver_used + ' - Case: ' + case.name)
    fig.add_vline(guessed_changepoints, line_width=3, line_color='red')
    fig.show()


def visualize_silhouette(case: Case, solution: Solution, median_silhouettes: List[float], guessed_changepoints: int):
    """
    Visualizes a single case median silhouettes for solutions
     with different amount of changepoints.
    :param median_silhouettes: A list with the median silhouette value for each solution of each changepoint.
    :return: None.
    """
    amount_changepoints = len(median_silhouettes)
    df_rows = [[k + 1, median_silhouettes[k]] for k in range(amount_changepoints)]
    df = pd.DataFrame(df_rows, columns=['changepoints', 'aggregated silhouette'])
    fig = px.line(df, x='changepoints', y='aggregated silhouette', title='Silhouette - ' + solution.metrics.solver_used + ' - Case: ' + case.name)
    fig.add_vline(guessed_changepoints, line_width=3, line_color='red')
    fig.show()
