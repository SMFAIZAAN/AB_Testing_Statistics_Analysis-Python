# 🛒 A/B Testing — ShopEasy Checkout Page Redesign

_Full end-to-end A/B test measuring the impact of a redesigned checkout page on conversion rate and revenue using Python — covering EDA, hypothesis testing, effect size, confidence intervals, and statistical power analysis across 10,000 users._

---

## 📌 Table of Contents
- <a href="#overview">Overview</a>
- <a href="#business-problem">Business Problem</a>
- <a href="#dataset">Dataset</a>
- <a href="#tools--technologies">Tools & Technologies</a>
- <a href="#project-structure">Project Structure</a>
- <a href="#data-cleaning--preparation">Data Cleaning & Preparation</a>
- <a href="#exploratory-data-analysis-eda">Exploratory Data Analysis (EDA)</a>
- <a href="#research-questions--key-findings">Research Questions & Key Findings</a>
- <a href="#dashboard">Dashboard</a>
- <a href="#how-to-run-this-project">How to Run This Project</a>
- <a href="#final-recommendations">Final Recommendations</a>

---

<h2><a class="anchor" id="overview"></a>Overview</h2>

This project runs a controlled A/B experiment to test whether a redesigned checkout page increases conversion rate and revenue per user on a fictional e-commerce platform, ShopEasy. The test ran across 10,000 users — 5,000 control, 5,000 treatment. The new page delivered a +3.18 percentage point conversion lift and +$3.02 revenue per user, both statistically significant at p < 0.0001 with 99.9% statistical power. The analysis covers EDA, normality checks, two-proportion Z-test, Welch's t-test, Cohen's effect size, 95% confidence intervals, and power analysis.

---

<h2><a class="anchor" id="business-problem"></a>Business Problem</h2>

ShopEasy's product team hypothesized that a streamlined checkout page would reduce drop-offs and increase completed purchases. Before rolling it out to all users, a controlled experiment was run to validate the hypothesis with statistical rigor.

The new checkout page resulted in:
- Conversion rate: **9.58% → 12.76%** (+3.18pp absolute, +33.2% relative lift, p = 0.0000)
- Revenue per user: **$8.23 → $11.24** (+$3.02, +36.6% relative lift, p = 0.0000)
- Projected annual revenue uplift: **~$335,217**

**Core question:** Does the new checkout page design significantly increase conversion rate and revenue per user compared to the existing design?

---

<h2><a class="anchor" id="dataset"></a>Dataset</h2>

- **Source:** Simulated e-commerce dataset (Python-generated)
- **Size:** 10,000 users — 5,000 control, 5,000 treatment (perfectly balanced)
- **Key columns:**

| Column | Description |
|---|---|
| `user_id` | Unique identifier per user |
| `group` | Control or Treatment |
| `converted` | Binary outcome — 1 = purchased, 0 = did not |
| `order_value` | Value of the order in USD |
| `revenue_per_user` | Total revenue attributed per user in USD |
| `session_duration` | Time spent on page in seconds |

---

<h2><a class="anchor" id="tools--technologies"></a>Tools & Technologies</h2>

- Python (Pandas, NumPy, Matplotlib, Seaborn, SciPy, Statsmodels)
- Jupyter Notebook
- GitHub

---

<h2><a class="anchor" id="project-structure"></a>Project Structure</h2>

```
ab-testing-shopeasy/
│
├── README.md
├── requirements.txt
├── AB_Testing_Report.pdf
│
├── notebooks/
│   └── ab_testing_analysis.ipynb     # Full analysis pipeline
│
├── data/
│   └── ab_test_data.csv              # Simulated dataset (10,000 users)
│
├── images/
│   ├── 01_eda_overview.png
│   ├── 02_key_metrics.png
│   ├── 03_normality_checks.png
│   ├── 04_significance.png
│   ├── 05_effect_size.png
│   ├── 06_confidence_intervals.png
│   ├── 07_power_curve.png
│   └── 08_final_dashboard.png
```

---

<h2><a class="anchor" id="data-cleaning--preparation"></a>Data Cleaning & Preparation</h2>

- Verified group balance: exactly 5,000 users per group
- Checked for duplicate `user_id` entries — none found
- Validated `converted` column contains only binary values (0 or 1)
- Confirmed no missing values across all key columns
- Inspected `order_value` for outliers using IQR method
- Split into separate `df_control` and `df_treatment` DataFrames for isolated analysis

---

<h2><a class="anchor" id="exploratory-data-analysis-eda"></a>Exploratory Data Analysis (EDA)</h2>

**Group Balance Check:**
- Control: 5,000 users | Treatment: 5,000 users — perfectly balanced split
- Control: 479 conversions = 9.58% | Treatment: 638 conversions = 12.76%

![EDA Group Overview](images/01_eda_overview.png)

**Key Metrics Comparison:**
- Conversion Rate: 9.58% (control) vs 12.76% (treatment)
- Revenue per User: $8.23 (control) vs $11.24 (treatment)
- Avg Session Duration: 119s (control) vs 104s (treatment) — shorter sessions in treatment suggest a more frictionless checkout flow

![Key Metrics](images/02_key_metrics.png)

**Normality Check — Order Value (Buyers Only):**
- Control mean order value: $85.9 | Treatment mean order value: $88.1
- Histograms and Q-Q plots confirmed approximately normal distributions for buyers in both groups — justifying use of Welch's t-test for revenue comparison

![Normality Checks](images/03_normality_checks.png)

---

<h2><a class="anchor" id="research-questions--key-findings"></a>Research Questions & Key Findings</h2>

### 1. Conversion Rate — Two-Proportion Z-Test

```python
from statsmodels.stats.proportion import proportions_ztest

counts = [df_control['converted'].sum(), df_treatment['converted'].sum()]
nobs   = [len(df_control), len(df_treatment)]

z_stat, p_value = proportions_ztest(counts, nobs)
# z = 5.05  |  p = 0.0000  →  SIGNIFICANT
```

**Answers:** Is the treatment conversion rate (12.76%) meaningfully higher than control (9.58%)?  
→ Yes. +3.18 percentage points absolute, +33.2% relative lift.

**95% CI — Conversion Rate Difference:** [+2.0%, +4.4%] — excludes 0 ✅

![Confidence Intervals](images/06_confidence_intervals.png)

---

### 2. Revenue per User — Welch's t-Test

```python
from scipy.stats import ttest_ind

t_stat, p_value = ttest_ind(
    df_control['revenue_per_user'],
    df_treatment['revenue_per_user'],
    equal_var=False   # Welch's — no assumption of equal variance
)
# t = 5.27  |  p = 0.0000  →  SIGNIFICANT
```

**Answers:** Did the new page drive more revenue per user ($11.24 vs $8.23)?  
→ Yes. +$3.02 per user (p < 0.0001).

![Statistical Significance](images/04_significance.png)

---

### 3. Effect Size — Cohen's h & Cohen's d

```python
from statsmodels.stats.proportion import proportion_effectsize

cohen_h = proportion_effectsize(p_treatment, p_control)
# h = 0.101  →  Negligible effect (benchmark: small = 0.2)

pooled_std = np.sqrt((std_c**2 + std_t**2) / 2)
cohen_d = (mean_treatment - mean_control) / pooled_std
# d = 0.105  →  Negligible effect
```

**Answers:** How large is the practical impact?  
→ Both metrics are statistically significant but show negligible effect size. At n = 10,000, even small true differences will hit significance. Statistical significance does not equal practical significance.

![Effect Size](images/05_effect_size.png)

---

### 4. Statistical Power Analysis

```python
from statsmodels.stats.power import NormalIndPower

analysis = NormalIndPower()
power = analysis.solve_power(effect_size=cohen_h, nobs1=5000, alpha=0.05)
# Power = 99.9%  |  Min n needed = 1,533 per group
```

**Answers:** Was the study adequately powered?  
→ Yes — well above the 80% threshold. The study was over-powered; only 1,533 users per group were needed to detect this effect. The large sample size is partly why a negligible effect still reached significance.

![Power Curve](images/07_power_curve.png)

---

<h2><a class="anchor" id="dashboard"></a>Dashboard</h2>

The final results dashboard consolidates conversion rates, revenue per user, session duration, 95% CI, and the power curve in a single view.

![Final Dashboard](images/08_final_dashboard.png)

---

<h2><a class="anchor" id="how-to-run-this-project"></a>How to Run This Project</h2>

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ab-testing-shopeasy.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Open and run the notebook:
```bash
jupyter notebook notebooks/ab_testing_analysis.ipynb
```

---

<h2><a class="anchor" id="final-recommendations"></a>Final Recommendations</h2>

**Summary of results:**

| Metric | Control | Treatment | Δ Absolute | Δ Relative | p-value |
|---|---|---|---|---|---|
| Conversion Rate | 9.58% | 12.76% | +3.18pp | +33.2% | 0.0000 ✅ |
| Revenue per User | $8.23 | $11.24 | +$3.02 | +36.7% | 0.0000 ✅ |
| Avg Session Duration | 119s | 104s | −15s | −12.6% | — |

**Verdict: Ship the new checkout page.** Projected annual revenue uplift: ~$335,217.

**Before full rollout:**
- Run segment analysis by device type and new vs. returning users — a 33% lift uniformly across all segments is worth scrutinising
- Plot conversion rate over time to check for a novelty effect; early spikes can inflate treatment results in short experiments
- Present Cohen's d (0.105) alongside p-values in stakeholder reports — "p = 0.000" alone can lead to over-interpretation
- Consider sequential testing (mSPRT) for future experiments to allow early stopping without inflating Type I error rate
