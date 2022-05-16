"""Compare several formulas for calculating
relative difference of two sample means.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from scipy import stats
from tqdm import tqdm


SEED = 42
TRUE_CONTROL_MEAN = 5
TRUE_RELATIVE_DIFF = 0.3
TRUE_TREATMENT_MEAN = TRUE_CONTROL_MEAN * (1 + TRUE_RELATIVE_DIFF)

CONTROL_SIZE = 1000
TREATEMT_SIZE = CONTROL_SIZE
N_EXPERIMENTS = 1000

SHAPE = 0.2
CONTROL_PARAMS = {
    'n': SHAPE,
    'p': SHAPE / (SHAPE + TRUE_CONTROL_MEAN),
}
TREATMENT_PARAMS = {
    'n': SHAPE,
    'p': SHAPE / (SHAPE + TRUE_TREATMENT_MEAN),
}


def calc_experiment(
    control_params: dict[str, float],
    treatment_params: dict[str, float],
    control_size: int,
    treatment_size: int,
    sample_distr: Callable,
    size_distr: Callable,
) -> dict[str, float]:
    control = sample_distr(
        **control_params,
        size=size_distr(control_size),
    )
    treatment = sample_distr(
        **treatment_params,
        size=size_distr(treatment_size),
    )

    mean1 = np.mean(treatment)
    mean2 = np.mean(control)
    mean1_sq = mean1 * mean1
    mean2_sq = mean2 * mean2
    n1 = len(treatment)
    n2 = len(control)
    vn1 = np.var(treatment, ddof=1) / n1
    vn2 = np.var(control, ddof=1) / n2
    ratio = mean1 / mean2
    rel_diff = ratio - 1

    abs_ppf = stats.t.ppf(
        q=0.975,
        df=(vn1 + vn2)**2 / (vn1**2 / (n1 - 1) + vn2**2 / (n2 - 1)),
    )
    naive_lower, naive_upper = (
        rel_diff
        + np.array([-abs_ppf, abs_ppf]) * np.sqrt(vn1 + vn2) / mean2
    )

    cv1_sq = vn1 / mean1_sq
    cv2_sq = vn2 / mean2_sq
    scale = np.sqrt(cv1_sq + cv2_sq)
    rel_ppf = stats.t.ppf(
        q=0.975,
        df=(cv1_sq + cv2_sq)**2 / (cv1_sq**2 / (n1 - 1) + cv2_sq**2 / (n2 - 1)),
    )
    t = np.array([-rel_ppf, rel_ppf])

    delta_lower, delta_upper = ratio * (1 + t * scale) - 1
    logdelta_lower, logdelta_upper = ratio * np.exp(t * scale) - 1

    rel_ppf_sq = rel_ppf * rel_ppf
    radic = cv1_sq + cv2_sq - rel_ppf_sq * cv1_sq * cv2_sq
    denom = 1 - rel_ppf_sq * cv2_sq
    fieller_lower, fieller_upper = ratio * (1 + t * np.sqrt(radic)) / denom - 1

    return {
        'naive_lower': naive_lower,
        'naive_upper': naive_upper,
        'delta_lower': delta_lower,
        'delta_upper': delta_upper,
        'logdelta_lower': logdelta_lower,
        'logdelta_upper': logdelta_upper,
        'fieller_lower': fieller_lower,
        'fieller_upper': fieller_upper,
    }


def null_rejected(x: ArrayLike) -> dict[str, float]:
    lower, upper = (
        stats.binomtest(k=np.sum(x), n=len(x), p=0.05)
        .proportion_ci(method='wilsoncc')
    )

    return {
        'mean': np.mean(x),
        'conf_int': f'({round(lower, 4)}, {round(upper, 4)})',
    }


def true_param_location(
    lower: ArrayLike,
    upper: ArrayLike,
    true_param_value: float,
) -> dict[str, float]:
    x = (lower < true_param_value) & (true_param_value < upper)
    x_lower, x_upper = (
        stats.binomtest(k=np.sum(x), n=len(x), p=0.05)
        .proportion_ci(method='wilsoncc')
    )

    return {
        'conf_level': x.mean(),
        'conf_level_ci': f'({round(x_lower, 4)}, {round(x_upper, 4)})',
        'quantiles': (
            f'({np.mean(true_param_value <= lower)}'
            f', {np.mean(true_param_value < upper)})'
        ),
    }


def main():
    rng = np.random.default_rng(SEED)

    aa_data = pd.DataFrame(
        calc_experiment(
            control_params=CONTROL_PARAMS,
            treatment_params=CONTROL_PARAMS,
            control_size=CONTROL_SIZE,
            treatment_size=TREATEMT_SIZE,
            sample_distr=rng.negative_binomial,
            size_distr=rng.poisson,
        )
        for i in tqdm(range(N_EXPERIMENTS))
    )

    print(
        pd.DataFrame({
            ci: null_rejected(
                aa_data[f'{ci}_lower'].ge(0) | aa_data[f'{ci}_upper'].le(0)
            )
            for ci in ['naive', 'delta', 'logdelta', 'fieller']
        })
        .transpose()
    )

    print(
        pd.DataFrame({
            ci: true_param_location(
                lower=aa_data[f'{ci}_lower'],
                upper=aa_data[f'{ci}_upper'],
                true_param_value=0,
            )
            for ci in ['naive', 'delta', 'logdelta', 'fieller']
        })
        .transpose()
    )


    ab_data = pd.DataFrame(
        calc_experiment(
            control_params=CONTROL_PARAMS,
            treatment_params=TREATMENT_PARAMS,
            control_size=CONTROL_SIZE,
            treatment_size=TREATEMT_SIZE,
            sample_distr=rng.negative_binomial,
            size_distr=rng.poisson,
        )
        for i in tqdm(range(N_EXPERIMENTS))
    )

    print(
        pd.DataFrame({
            ci: null_rejected(
                ab_data[f'{ci}_lower'].ge(0) | aa_data[f'{ci}_upper'].le(0)
            )
            for ci in ['naive', 'delta', 'logdelta', 'fieller']
        })
        .transpose()
    )

    print(
        pd.DataFrame({
            ci: true_param_location(
                lower=ab_data[f'{ci}_lower'],
                upper=ab_data[f'{ci}_upper'],
                true_param_value=TRUE_RELATIVE_DIFF,
            )
            for ci in ['naive', 'delta', 'logdelta', 'fieller']
        })
        .transpose()
    )

if __name__ == '__main__':
    main()
