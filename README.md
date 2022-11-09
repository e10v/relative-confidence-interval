Compare several ways for calculating confidence interval (CI) of the relative difference of two sample means:
- naive (and also wrong) -- absolute CI divided by control mean.
- delta -- relative CI with delta method.
- logdelta -- also delta method, only applied to log-transformed ratio.
- fieller -- Fieller's confidence interval.

The script does the following:
- Simulates many experiments to generate pairs of samples (AA or AB).
- Calculates proportions of samples where does not include zero (type I error or statistical power).
- Calculates true confidence level and it's quantiles (since we know the true parameter value)

Links:
- https://en.wikipedia.org/wiki/Delta_method
- https://en.wikipedia.org/wiki/Fieller%27s_theorem
- https://github.com/cran/mratios/blob/master/R/ttestratio.R
