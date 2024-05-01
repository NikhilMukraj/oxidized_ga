from oxidized_ga import oxidized_ga
from concurrent.futures import ThreadPoolExecutor
import numpy as np


def parallel_wrapper(f, workers, pop, args):
    result = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i in pop:
            result.append(executor.submit(f, i, *args))

    return [i.result() for i in result]

def wrapped_genetic_algo(
    objective_func,
    lower_bounds,
    upper_bounds,
    n_bits,
    n_iter,
    n_pop,
    r_cross,
    r_mut,
    k,
    *,
    settings={},
    workers=0,
    total_n_bits=None,
    print_output=None
):
    if not isinstance(workers, int) or workers < 0:
        raise ValueError("parallel argument must be integer greater than or equal to 0 or None")

    if workers > 0:
        output = oxidized_ga.genetic_algo(
            objective_func,
            lower_bounds,
            upper_bounds,
            n_bits,
            n_iter,
            n_pop,
            r_cross,
            r_mut,
            k,
            settings,
            (parallel_wrapper, workers),
            total_n_bits,
            print_output
        )
    else:
        output = oxidized_ga.genetic_algo(
            objective_func,
            lower_bounds,
            upper_bounds,
            n_bits,
            n_iter,
            n_pop,
            r_cross,
            r_mut,
            k,
            settings,
            None,
            total_n_bits,
            print_output
        )

    return output


def check_architecture_type(arch):
    if not isinstance(arch, dict):
        raise TypeError('architecture must be a dictionary')

    for key, value in arch.items():
        if any([not isinstance(i, oxidized_ga.Activation) for i in value[0]]):
            raise ValueError('first element of each layer must be list of activations')
        if len(value[1]) != 2:
            raise ValueError('second element of each layer must be a list of length 2')
        if any([not isinstance(i, int) for i in value[1]]):
            raise ValueError('second element of each layer must be list of floats')

def neural_net_genetic_algo(
    objective_func,
    lower_bounds,
    upper_bounds,
    n_bits,
    activation_precision,
    arch,
    n_iter,
    n_pop,
    r_cross,
    r_mut,
    k,
    *,
    settings={},
    workers=0,
    print_output=None
):
    if not isinstance(lower_bounds, float) and not isinstance(lower_bounds, int):
        raise TypeError('lower_bounds must be a numeric')
    if not isinstance(upper_bounds, float) and not isinstance(upper_bounds, int):
        raise TypeError('upper_bounds must be a numeric')

    check_architecture_type(arch)

    neuron_len, activation_len = oxidized_ga.neural_network_get_bits(n_bits, activation_precision, arch)
    lower_bounds = np.array([lower_bounds for i in range(neuron_len)], dtype=np.float32)
    upper_bounds = np.array([upper_bounds for i in range(neuron_len)], dtype=np.float32)

    total_n_bits = neuron_len + activation_len

    return wrapped_genetic_algo(
            objective_func,
            lower_bounds,
            upper_bounds,
            n_bits,
            n_iter,
            n_pop,
            r_cross,
            r_mut,
            k,
            settings=settings,
            workers=workers,
            total_n_bits=total_n_bits,
            print_output=print_output
        )
