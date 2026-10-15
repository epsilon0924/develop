# Training Analysis

## Dataset

- Source: `logs/result.csv`
- Configuration: `config.json`
- Rows: **5,023**
- Columns: **23**
- Episode range: **1 to 5,019**
- Latest-window size: **100**

## Evaluation Summary

| Metric | Overall | Latest Window |
| --- | --- | --- |
| Total score | 27.412 | 92.425 |
| Success rate | 9.178% | 94.000% |
| Episode duration | 37.586% | 97.532% |
| Duration consistency | 31.914% | 87.721% |
| Force efficiency | 72.191% | 77.068% |

Scores range from 0 to 100. Higher values indicate better performance according to the configured evaluation function.

## Score Contributions

| Component | Weight | Overall Score | Latest-Window Score |
| --- | --- | --- | --- |
| success_rate | 45.000% | 4.130 | 42.300 |
| episode_duration | 30.000% | 11.276 | 29.260 |
| duration_consistency | 15.000% | 4.787 | 13.158 |
| force_efficiency | 10.000% | 7.219 | 7.707 |

## Result Distribution

| Result | Count | Rate |
| --- | --- | --- |
| fallen | 2,619 | 52.140% |
| outside_track | 1,943 | 38.682% |
| success | 461 | 9.178% |

## Overall Statistics

| Metric | Count | Mean | Std | Min | Q1 | Median | Q3 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| steps | 5,023 | 187.929 | 136.040 | 8.000 | 108.000 | 159.000 | 235.500 | 500.000 |
| total_reward | 5,023 | 169.689 | 137.625 | -4.403 | 91.681 | 138.384 | 213.009 | 509.323 |
| initial_cart_position | 5,023 | 0.001 | 0.029 | -0.050 | -0.024 | 0.001 | 0.026 | 0.050 |
| initial_cart_velocity | 5,023 | -0.000 | 0.029 | -0.050 | -0.025 | -0.000 | 0.024 | 0.050 |
| initial_angle_degrees | 5,023 | 0.025 | 1.645 | -2.864 | -1.394 | 0.004 | 1.441 | 2.864 |
| initial_angular_velocity | 5,023 | -0.000 | 0.029 | -0.050 | -0.025 | -0.001 | 0.024 | 0.050 |
| final_cart_position | 5,023 | 1.194 | 1.456 | -2.439 | 0.068 | 2.049 | 2.406 | 2.445 |
| final_cart_velocity | 5,023 | 0.823 | 1.077 | -2.927 | 0.051 | 1.083 | 1.674 | 2.732 |
| final_angle_degrees | 5,023 | 4.126 | 8.987 | -14.231 | -1.770 | 8.226 | 12.166 | 14.139 |
| final_angular_velocity | 5,023 | 0.115 | 0.543 | -2.270 | -0.076 | 0.144 | 0.397 | 2.056 |
| mean_force | 5,023 | 0.293 | 0.714 | -5.574 | 0.008 | 0.377 | 0.659 | 4.741 |
| mean_absolute_force | 5,023 | 2.781 | 1.579 | 0.788 | 1.488 | 1.896 | 4.392 | 7.008 |
| maximum_absolute_force | 5,023 | 8.241 | 1.607 | 3.000 | 7.083 | 8.823 | 9.629 | 10.000 |
| environment_steps_per_second | 5,023 | 516.488 | 94.862 | 3.900 | 501.292 | 561.137 | 566.727 | 569.123 |

## First Window vs. Latest Window

The first and latest windows each contain up to **100 episodes**.

| Metric | First Window | Latest Window | Relative Change |
| --- | --- | --- | --- |
| steps | 26.070 | 487.660 | 1770.579% |
| total_reward | 12.647 | 481.306 | 3705.711% |
| mean_absolute_force | 5.247 | 2.293 | -56.292% |
| maximum_absolute_force | 9.580 | 7.400 | -22.760% |
| environment_steps_per_second | 229.733 | 567.842 | 147.174% |
| success_rate | 0.000% | 94.000% | 94.000% |

For the success rate, the change is reported in percentage points. For other metrics, relative change is calculated against the absolute first-window mean.

## Correlation Matrix

| Metric | steps | total_reward | initial_angle_degrees | initial_cart_velocity | initial_angular_velocity | mean_force | mean_absolute_force | maximum_absolute_force |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| steps | 1.000 | 0.998 | -0.056 | -0.030 | -0.004 | -0.097 | -0.509 | -0.296 |
| total_reward | 0.998 | 1.000 | -0.056 | -0.029 | -0.004 | -0.103 | -0.505 | -0.314 |
| initial_angle_degrees | -0.056 | -0.056 | 1.000 | -0.013 | 0.015 | 0.137 | -0.006 | -0.024 |
| initial_cart_velocity | -0.030 | -0.029 | -0.013 | 1.000 | 0.020 | 0.016 | 0.033 | 0.021 |
| initial_angular_velocity | -0.004 | -0.004 | 0.015 | 0.020 | 1.000 | 0.029 | -0.016 | -0.042 |
| mean_force | -0.097 | -0.103 | 0.137 | 0.016 | 0.029 | 1.000 | -0.194 | -0.175 |
| mean_absolute_force | -0.509 | -0.505 | -0.006 | 0.033 | -0.016 | -0.194 | 1.000 | 0.682 |
| maximum_absolute_force | -0.296 | -0.314 | -0.024 | 0.021 | -0.042 | -0.175 | 0.682 | 1.000 |

Correlations describe linear associations and do not establish causation.

## Evaluation Function

The total evaluation value is defined as:

$$
E = 100 \left(0.450000S + 0.300000D + 0.150000C + 0.100000F\right)
$$

where:

$$
S = \frac{N_{success}}{N}
$$

$$
D = \frac{1}{N}\sum_{i=1}^{N}\min\left(\frac{t_i}{T}, 1\right)
$$

$$
RMSE_t = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\left(T - \min(t_i,T)\right)^2}
$$

$$
C = \max\left(0,1 - \frac{RMSE_t}{T}\right)
$$

$$
F = \max\left(0,1 - \frac{1}{N}\sum_{i=1}^{N}\min\left(\frac{|\bar{f}_i|}{f_{max}}, 1\right)\right)
$$

Definitions:

- $E$: total evaluation value from 0 to 100
- $N$: number of analyzed episodes
- $N_{success}$: number of successful episodes
- $t_i$: number of steps in episode $i$
- $T$: target number of steps
- $\bar{f}_i$: mean absolute force in episode $i$
- $f_{max}$: configured maximum force
- $S$: success component
- $D$: episode-duration component
- $C$: duration-consistency component
- $F$: force-efficiency component

The score is a project-specific measurement. It is not a standardized reinforcement-learning benchmark.

