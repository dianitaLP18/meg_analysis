| Setting       | Model               | intra-test   | cross-test1   | cross-test2   | cross-test3   |   Mean |
|:--------------|:--------------------|:-------------|:--------------|:--------------|:--------------|-------:|
| Intra-subject | majority            | 25.00        | nan           | nan           | nan           | nan    |
| Intra-subject | logistic_regression | 60.29        | nan           | nan           | nan           | nan    |
| Intra-subject | cnn                 | 61.58 ± 5.46 | nan           | nan           | nan           | nan    |
| Intra-subject | eegnet              | 69.98 ± 9.22 | nan           | nan           | nan           | nan    |
| Cross-subject | majority            | nan          | 25.00         | 25.00         | 25.00         | nan    |
| Cross-subject | logistic_regression | nan          | 60.11         | 34.38         | 44.30         | nan    |
| Cross-subject | cnn                 | nan          | 41.30 ± 1.60  | 43.69 ± 4.81  | 42.49 ± 0.84  |  42.49 |
| Cross-subject | eegnet              | nan          | 54.14 ± 3.91  | 35.17 ± 4.00  | 44.33 ± 1.50  |  44.55 |