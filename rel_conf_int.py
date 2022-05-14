from collections.abc import Callable
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from IPython.display import display


seed = 42
true_control_mean = 5
true_relative_diff = 0.2
shape = 0.2
control_size = 2000
treatment_size = control_size
n_experiments = 10000


true_treatment_mean = true_control_mean * (1 + true_relative_diff)
control_params = {"n": shape, "p": shape / (shape + true_control_mean)}
treatment_params = {"n": shape, "p": shape / (shape + true_treatment_mean)}

rng = np.random.default_rng(seed)
sample_distr = rng.negative_binomial
size_distr = rng.poisson


def calc_experiment(
    control_params: dict[str, float],
    treatment_params: dict[str, float],
    control_size: int,
    treatment_size: int,
    sample_distr: Callable,
    size_distr: Callable,
) -> dict[str, float]:
    control = sample_distr(**control_params, size=size_distr(control_size))
    treatment = sample_distr(**treatment_params, size=size_distr(treatment_size))

    mean1 = np.mean(treatment)
    mean2 = np.mean(control)
    mean1_sq = mean1 * mean1
    mean2_sq = mean2 * mean2
    n1 = len(treatment)
    n2 = len(control)
    vn1 = np.var(treatment, ddof=1) / n1
    vn2 = np.var(control, ddof=1) / n2
    rel_diff = mean1 / mean2 - 1

    abs_pvalue = stats.ttest_ind(treatment, control, equal_var="False").pvalue
    abs_t = stats.t((vn1 + vn2) ** 2 / (vn1**2 / (n1 - 1) + vn2**2 / (n2 - 1)))
    naive_lower, naive_upper = (
        rel_diff + abs_t.ppf([0.025, 0.975]) * np.sqrt(vn1 + vn2) / mean2
    )

    delta_norm = stats.norm(
        loc=rel_diff,
        scale=np.sqrt((vn1 + vn2 * mean1_sq / mean2_sq) / mean2_sq),
    )
    delta_pvalue = 2 * min(delta_norm.cdf(0), delta_norm.sf(0))
    delta_lower, delta_upper = delta_norm.ppf([0.025, 0.975])

    cv1_sq = vn1 / mean1_sq
    cv2_sq = vn2 / mean2_sq
    fieller_ppf = stats.t.ppf(
        0.975,
        (cv1_sq + cv2_sq) ** 2 / (cv1_sq**2 / (n1 - 1) + cv2_sq**2 / (n2 - 1)),
    )
    fieller_ppf_sq = fieller_ppf * fieller_ppf
    fieller_lower, fieller_upper = (
        (mean1 / mean2)
        * (
            1
            + np.array([-fieller_ppf, fieller_ppf])
            * np.sqrt(cv1_sq + cv2_sq - fieller_ppf_sq * cv1_sq * cv2_sq)
        )
        / (1 - fieller_ppf_sq * cv2_sq)
    ) - 1

    return {
        "mean1": mean1,
        "mean2": mean2,
        "rel_diff": rel_diff,
        "abs_pvalue": abs_pvalue,
        "naive_lower": naive_lower,
        "naive_upper": naive_upper,
        "delta_pvalue": delta_pvalue,
        "delta_lower": delta_lower,
        "delta_upper": delta_upper,
        "fieller_lower": fieller_lower,
        "fieller_upper": fieller_upper,
    }


def null_rejected(x: pd.Series) -> dict[str, float]:
    lower, upper = stats.binomtest(k=np.sum(x), n=len(x), p=0.05).proportion_ci(
        method="wilsoncc"
    )
    return {"mean": np.mean(x), "conf_int": f"({round(lower, 4)}, {round(upper, 4)})"}


def true_param_location(
    lower: pd.Series, upper: pd.Series, true_param_value: float
) -> dict[str, float]:
    x = lower.lt(true_param_value) & upper.gt(true_param_value)
    x_lower, x_upper = stats.binomtest(k=np.sum(x), n=len(x), p=0.05).proportion_ci(
        method="wilsoncc"
    )
    return {
        # "conf_level": (lower.lt(true_param_value) & upper.gt(true_param_value)).mean(),
        "conf_level": x.mean(),
        "conf_level_ci": f"({round(x_lower, 4)}, {round(x_upper, 4)})",
        "quantiles": f"({lower.ge(true_param_value).mean()}, {upper.gt(true_param_value).mean()})",
    }


aa_data = pd.DataFrame(
    calc_experiment(
        control_params=control_params,
        treatment_params=control_params,
        control_size=control_size,
        treatment_size=treatment_size,
        sample_distr=sample_distr,
        size_distr=size_distr,
    )
    for i in tqdm(range(n_experiments))
)


display(
    pd.DataFrame(
        {
            pvalue_col: null_rejected(aa_data[pvalue_col].le(0.05))
            for pvalue_col in ["abs_pvalue", "delta_pvalue"]
        }
    ).transpose()
)

display(
    pd.DataFrame(
        {
            prefix: null_rejected(
                aa_data[f"{prefix}_lower"].ge(0) | aa_data[f"{prefix}_upper"].le(0)
            )
            for prefix in ["naive", "delta", "fieller"]
        }
    ).transpose()
)

display(
    pd.DataFrame(
        {
            prefix: true_param_location(
                lower=aa_data[f"{prefix}_lower"],
                upper=aa_data[f"{prefix}_upper"],
                true_param_value=0,
            )
            for prefix in ["naive", "delta", "fieller"]
        }
    ).transpose()
)


ab_data = pd.DataFrame(
    calc_experiment(
        control_params=control_params,
        treatment_params=treatment_params,
        control_size=control_size,
        treatment_size=treatment_size,
        sample_distr=sample_distr,
        size_distr=size_distr,
    )
    for i in tqdm(range(n_experiments))
)

display(
    pd.DataFrame(
        {
            pvalue_col: null_rejected(ab_data[pvalue_col].le(0.05))
            for pvalue_col in ["abs_pvalue", "delta_pvalue"]
        }
    ).transpose()
)

display(
    pd.DataFrame(
        {
            prefix: null_rejected(
                ab_data[f"{prefix}_lower"].ge(0) | ab_data[f"{prefix}_upper"].le(0)
            )
            for prefix in ["naive", "delta", "fieller"]
        }
    ).transpose()
)

display(
    pd.DataFrame(
        {
            prefix: true_param_location(
                lower=ab_data[f"{prefix}_lower"],
                upper=ab_data[f"{prefix}_upper"],
                true_param_value=true_relative_diff,
            )
            for prefix in ["naive", "delta", "fieller"]
        }
    ).transpose()
)
