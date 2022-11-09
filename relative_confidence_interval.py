"""Compares several ways for calculating confidence interval (CI)
of the relative difference of two sample means:
- naive (and also wrong) -- absolute CI divided by control mean.
- delta -- relative CI with delta method.
- logdelta -- also delta method, only applied to log-transformed ratio.
- fieller -- Fieller's confidence interval.

The script does the following:
- Simulates many experiments to generate pairs of samples (AA or AB).
- Calculates proportions of samples where CI does not include zero
(type I error or statistical power).
- Calculates true confidence level and it's quantiles
(since we know the true parameter value).

Usage:
```
python relative_confidence_interval.py > results.txt
```

Links:
- https://en.wikipedia.org/wiki/Delta_method
- https://en.wikipedia.org/wiki/Fieller%27s_theorem
- https://github.com/cran/mratios/blob/master/R/ttestratio.R
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from scipy import stats
import tqdm


RANDOM_SEED = 42
N_EXPERIMENTS = 10000
CONTROL_SIZE = 1000
TREATMENT_SIZE = CONTROL_SIZE

TRUE_CONTROL_MEAN = 5
TRUE_RELATIVE_DIFF = 0.3
TRUE_TREATMENT_MEAN = TRUE_CONTROL_MEAN * (1 + TRUE_RELATIVE_DIFF)

# Sample from skewed negative binomial distribution.
SAMPLE_DISTR = 'negative_binomial'
SHAPE = 0.2
SAMPLE_CONTROL_PARAMS = {
    'n': SHAPE,
    'p': SHAPE / (SHAPE + TRUE_CONTROL_MEAN),
}
SAMPLE_TREATMENT_PARAMS = {
    'n': SHAPE,
    'p': SHAPE / (SHAPE + TRUE_TREATMENT_MEAN),
}

# Sample sizes from poisson distibution.
SIZE_DISTR = 'poisson'
SIZE_CONTROL_PARAMS = {'lam': CONTROL_SIZE}
SIZE_TREATMENT_PARAMS = {'lam': TREATMENT_SIZE}


def calc_experiment(
    sample_distr: Callable,
    sample_control_params: dict,
    sample_treatment_params: dict,
    size_distr: Callable,
    size_control_params: dict,
    size_treatment_params: dict,
) -> dict:
    control = sample_distr(
        **sample_control_params,
        size=size_distr(**size_control_params),
    )
    treatment = sample_distr(
        **sample_treatment_params,
        size=size_distr(**size_treatment_params),
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


def null_rejected(
    lower: ArrayLike,
    upper: ArrayLike,
) -> dict:
    x = (lower > 0) | (upper < 0)
    x_lower, x_upper = (
        stats.binomtest(k=np.sum(x), n=len(x), p=0.05)
        .proportion_ci(method='wilsoncc')
    )

    return {
        'mean': np.mean(x),
        'conf_int': f'({round(x_lower, 4)}, {round(x_upper, 4)})',
    }


def true_param_location(
    lower: ArrayLike,
    upper: ArrayLike,
    true_param_value: float,
) -> dict:
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


if __name__ == '__main__':
    rng = np.random.default_rng(RANDOM_SEED)

    print('AA experiments:')
    aa_data = pd.DataFrame(
        calc_experiment(
            sample_distr=getattr(rng, SAMPLE_DISTR),
            sample_control_params=SAMPLE_CONTROL_PARAMS,
            sample_treatment_params=SAMPLE_CONTROL_PARAMS,
            size_distr=getattr(rng, SIZE_DISTR),
            size_control_params=SIZE_CONTROL_PARAMS,
            size_treatment_params=SIZE_TREATMENT_PARAMS,
        )
        for i in tqdm.tqdm(range(N_EXPERIMENTS))
    )
    print(aa_data.to_string(max_rows=10))

    print('\nType I error:')
    print(
        pd.DataFrame({
            ci: null_rejected(
                lower=aa_data[f'{ci}_lower'],
                upper=aa_data[f'{ci}_upper'],
            )
            for ci in ['naive', 'delta', 'logdelta', 'fieller']
        })
        .transpose()
    )

    print('\nTrue confidence levels:')
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


    print('\nAB experiments:')
    ab_data = pd.DataFrame(
        calc_experiment(
            sample_distr=getattr(rng, SAMPLE_DISTR),
            sample_control_params=SAMPLE_CONTROL_PARAMS,
            sample_treatment_params=SAMPLE_TREATMENT_PARAMS,
            size_distr=getattr(rng, SIZE_DISTR),
            size_control_params=SIZE_CONTROL_PARAMS,
            size_treatment_params=SIZE_TREATMENT_PARAMS,
        )
        for i in tqdm.tqdm(range(N_EXPERIMENTS))
    )
    print(ab_data.to_string(max_rows=10))

    print('\nStatistical power:')
    print(
        pd.DataFrame({
            ci: null_rejected(
                lower=ab_data[f'{ci}_lower'],
                upper=ab_data[f'{ci}_upper'],
            )
            for ci in ['naive', 'delta', 'logdelta', 'fieller']
        })
        .transpose()
    )

    print('\nTrue confidence levels:')
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
