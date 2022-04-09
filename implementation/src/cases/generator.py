import os
from typing import List, Tuple, Callable

import pandas as pd
import numpy as np
import math

from cases.case import CaseParameters
from utils.constants import Constants


def create_length_between_change_points(n: int, m: int, rng: np.random.RandomState) -> Tuple[List[int], List[int]]:
    """
    Generates a list of m integer changepoints in (0,n) and the space between them.
    :param n: Upper bound to generate changepoints.
    :param m: Amount of changepoints to be generated.
    :param rng: Fixed random number generator.
    :return: Two lists of size m indicating the space between changepoints and the inner changepoints.
    """
    change_points = sorted([0] + list(rng.choice(n, m, replace=False)) + [n])
    return list(np.diff(change_points)), \
           change_points[1:-1]


def gen_signal_change_mean(parameters: CaseParameters) -> Tuple[List[float], List[int]]:
    """
    Generates a signal that at each time it generates a random gaussian value sampled with N(mu, sigma) where
    mu is a value that is chosen uniformly between mu_low and mu_high at each changepoint.
    :param parameters: parameters that define the case.
    :return: A list of size n with the signal and the location of the m changepoints.
    """
    n, m, rng = parameters.size, parameters.changepoints, parameters.rng
    sigma, mu_low, mu_high = parameters.sigma, parameters.mu_low, parameters.mu_high
    diff, change_points = create_length_between_change_points(n, m, rng)
    return list(np.concatenate([rng.normal(rng.uniform(mu_low, mu_high), sigma, length) for length in diff])), \
           change_points


def gen_signal_change_variance(parameters: CaseParameters) -> Tuple[List[float], List[int]]:
    """
    Generates a signal that at each time it generates a random gaussian value sampled with N(mu, sigma) where
    sigma is a value that is chosen uniformly between sigma_low and sigma_high at each changepoint.
    :param parameters: parameters that define the case.
    :return: A list of size n with the signal and the location of the m changepoints.
    """
    n, m, rng = parameters.size, parameters.changepoints, parameters.rng
    mu, sigma_low, sigma_high = parameters.mu, parameters.sigma_low, parameters.sigma_high
    diff, change_points = create_length_between_change_points(n, m, rng)
    return list(np.concatenate([rng.normal(mu, rng.uniform(sigma_low, sigma_high), length) for length in diff])), \
           change_points


def gen_signal_exponential_change_mean(parameters: CaseParameters) -> Tuple[List[float], List[int]]:
    """
    Generates a signal that at each time it generates a random exponential value sampled with E(lambda) where
    lambda is a value that is chosen uniformly between lambda_low and lambda_high at each changepoint.
    :param parameters: parameters that define the case.
    :return: A list of size n with the signal and the location of the m changepoints.
    """
    n, m, rng = parameters.size, parameters.changepoints, parameters.rng
    lambda_low, lambda_high = parameters.lambda_low, parameters.lambda_high
    diff, change_points = create_length_between_change_points(n, m, rng)
    return list(np.concatenate([rng.exponential(rng.uniform(lambda_low, lambda_high), length) for length in diff])), \
           change_points


def gen_signal_dependant(parameters: CaseParameters) -> Tuple[List[float], List[int]]:
    """
    Generates a signal that at each time it generates a random exponential value sampled with N(mu, sigma) averaged
    with values that depend on previous values fo the signal, where mu is a value that is chosen uniformly between
    mu_low and mu_high at each changepoint.
    :param parameters: parameters that define the case.
    :return: A list of size n with the signal and the location of the m changepoints.
    """
    n, m, rng = parameters.size, parameters.changepoints, parameters.rng
    sigma, mu_low, mu_high = parameters.sigma, parameters.mu_low, parameters.mu_high
    diff, change_points = create_length_between_change_points(n, m, rng)
    alpha, beta = [0.5, 0.4, 0.099], 0.01
    signal = [0, 0, 0]
    diff[0] -= 3
    for length in diff:
        mu = rng.uniform(mu_low, mu_high)
        for i in range(length):
            signal.append((alpha[0] * signal[-1] +
                           alpha[1] * signal[-2] +
                           alpha[2] * signal[-3] +
                           beta * rng.normal(mu, sigma, 1)))
    return list(map(float, signal)), \
           change_points


def gen_signal(case_type: str) -> Callable[[CaseParameters], Tuple[List[float], List[int]]]:
    return {'mean': gen_signal_change_mean,
            'variance': gen_signal_change_variance,
            'exponential': gen_signal_exponential_change_mean,
            'dependant': gen_signal_dependant}[case_type]


def gen_real_signal(file_name: str = 'cardio', data_attribute: str = 'heartRate') -> List[float]:
    """
    It generates a real signal of all the stored values that we have, in order to be able to segment that later.
    :param file_name: File name where the table with values are stored.
    :param data_attribute: Column with the specific values that we want.
    :return: A list with the values retrieved in cronological order.
    """
    csv_file = pd.read_csv(''.join([Constants.project_root_path, 'resources/data/', file_name, '.csv']))
    return [x for x in csv_file[data_attribute]]


def write_csv(path: str, values: List[float]) -> None:
    """
    Auxiliary function used to write a list of values in a specific path.
    :param path: Path where we want to write the values.
    :param values: Values we want to write.
    :return: Nothing.
    """
    directory = '/'.join(path.split('/')[:-1])
    os.makedirs(directory, exist_ok=True)
    with open(path, 'w') as file_to_write:
        file_to_write.write(','.join(list(map(str, values))) + '\n')


def write_to_path(case_number: str, case_type: str, folder: str, extension: str, values: List[float]) -> None:
    """
    Auxiliar function used to write the values of a case in a specific path.
    :param case_number: Case number used to create the correct path.
    :param case_type: Case type used to create the correct path.
    :param folder: Folder where the case is to be saved.
    :param extension: Extension to be appended in the filename.
    :param values: Values to be saved.
    :return: Nothing.
    """
    path = ''.join([Constants.random_path, '/', folder, '/', case_number, '_', case_type, extension])
    write_csv(path, values)


def write_random_case(case_number: str, case_type: str, signal: List[float], change_points: List[int]) -> None:
    """
    Auxiliar function used to write the input case and its solution.
    :param case_number: Case number used to create the correct path.
    :param case_type: Case type used to create the correct path.
    :param signal: Values of the signal to be saved.
    :param change_points: Changepoints locations in the generated signal.
    :return: Nothing.
    """
    write_to_path(case_number, case_type, 'generated', '.in', signal)
    write_to_path(case_number, case_type, 'solutions', '.out', change_points)


def main() -> None:
    rng = np.random.RandomState(Constants.seed)
    case_sizes = [Constants.batch_size * (i + 1) for i in range(Constants.cases_per_type)]
    complete_real_signal = gen_real_signal('cardio', 'heartRate')
    for k, case_size in enumerate(case_sizes):
        case_number = str(k).zfill(2)

        mu_low, mu_high = rng.randint(-Constants.mean_limit, 0), rng.randint(0, Constants.mean_limit)
        sigma_low, sigma_high = 1, rng.randint(2, Constants.std_limit)
        lambda_low, lambda_high = 1, rng.randint(2, Constants.mean_limit)

        changepoint_base_amount = math.ceil(math.log(case_size))
        n, m = case_size, rng.randint(changepoint_base_amount // 2, 2 * changepoint_base_amount)
        mu, sigma = rng.randint(mu_low, mu_high), rng.randint(sigma_low, sigma_high)

        parameters = CaseParameters(n, m, mu, sigma, mu_low, mu_high, sigma_low, sigma_high, lambda_low, lambda_high)
        for case_type in ['mean', 'variance', 'exponential', 'dependant']:
            signal, change_points = gen_signal(case_type)(parameters)
            write_random_case(case_number, case_type, signal, change_points)

        real_signal_sample_length = int(Constants.min_days * Constants.minutes_in_a_day * ((k // 4) + 1))
        max_start_point = len(complete_real_signal) - real_signal_sample_length
        start = rng.randint(0, max_start_point)
        signal = complete_real_signal[start:(start + real_signal_sample_length)]
        write_csv(''.join([Constants.real_path, case_number, '_', 'real', '.in']), signal)


if __name__ == '__main__':
    main()
