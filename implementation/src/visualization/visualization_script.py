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
    :return: None
    """
    df = pd.DataFrame()
    is_random_signal = case.metadata[0].date == Constants.no_date
    label_x = 'index' if is_random_signal else 'date'
    label_y = 'value' if is_random_signal else 'BPM'
    for value, metadata in zip(case.signal, case.metadata):
        df = df.append({label_x: metadata.field_from_label(label_x),
                        label_y: value, },
                       ignore_index=True)
    fig = px.line(df, x=label_x, y=label_y)
    for changepoint in solution.changepoints:
        fig.add_vline(case.metadata[changepoint].field_from_label(label_x), line_width=3, line_color='red')

    fig.show()

    if __name__ == '__main__':
        pass
