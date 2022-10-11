import plotly.express as px
import pandas as pd

from cases.case import Case, ValueMetadata
from solution.solution import Solution
from utils.constants import Constants


def visualize(case: Case, solution: Solution) -> None:
    """
    Visualizes a single case and its solution.
    :param case: The input signal.
    :param solution: A solution with the changepoints found.
    :return: None.
    """
    is_random_signal = case.metadata[0].date == Constants.no_date
    label_x = 'index' if is_random_signal else 'date'
    label_y = 'value' if is_random_signal else 'BPM'
    df_rows = [[metadata.field_from_label(label_x), value] for value, metadata in zip(case.signal, case.metadata)]
    df = pd.DataFrame(df_rows, columns=[label_x, label_y])
    fig = px.line(df, x=label_x, y=label_y, title='Algorithm: ' + solution.metrics.solver_used + ' - Case: ' + case.name)
    for changepoint in solution.changepoints:
        fig.add_vline(case.metadata[changepoint].field_from_label(label_x), line_width=3, line_color='red')
    fig.show()


def visualize_elbow(case: Case, solution: Solution) -> None:
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
    fig.show()
