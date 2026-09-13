# ============================================================
# HYPOTHESIS TEST: Welch's t-test + 95% CI
# ============================================================
import numpy as np
from scipy import stats

promo = df[df.promo_flag == 1]["sales_total"]
nonpromo = df[df.promo_flag == 0]["sales_total"]

# --- Welch's t-test ---
t_stat, p_val = stats.ttest_ind(promo, nonpromo, equal_var=False)

# --- Difference in means + standard error ---
mean_diff = promo.mean() - nonpromo.mean()
se_diff = np.sqrt(promo.var(ddof=1)/len(promo) + nonpromo.var(ddof=1)/len(nonpromo))

# --- Welch-Satterthwaite degrees of freedom ---
df_welch = (promo.var(ddof=1)/len(promo) + nonpromo.var(ddof=1)/len(nonpromo))**2 / (
    (promo.var(ddof=1)/len(promo))**2 / (len(promo)-1) +
    (nonpromo.var(ddof=1)/len(nonpromo))**2 / (len(nonpromo)-1)
)

# --- 95% CI ---
t_crit = stats.t.ppf(0.975, df_welch)
ci_low  = mean_diff - t_crit * se_diff
ci_high = mean_diff + t_crit * se_diff

# --- Effect size (Cohen's d) ---
pooled_sd = np.sqrt((promo.var(ddof=1) + nonpromo.var(ddof=1)) / 2)
cohens_d = mean_diff / pooled_sd

# --- Robustness: Mann-Whitney U ---
u_stat, p_mw = stats.mannwhitneyu(promo, nonpromo, alternative="two-sided")

# --- Print results ---
print("=" * 60)
print("HYPOTHESIS TEST RESULTS")
print("=" * 60)
print(f"Promo mean:     ${promo.mean():,.2f}  (n={len(promo)})")
print(f"Non-promo mean: ${nonpromo.mean():,.2f}  (n={len(nonpromo)})")
print(f"Mean difference: ${mean_diff:,.2f}")
print(f"Standard error:  ${se_diff:,.2f}")
print(f"Welch t-stat:    {t_stat:.3f}")
print(f"Welch df:        {df_welch:.1f}")
print(f"p-value:         {p_val:.6f}")
print(f"95% CI:          [${ci_low:,.2f}, ${ci_high:,.2f}]")
print(f"Cohen's d:       {cohens_d:.3f}")
print(f"Mann-Whitney U:  {u_stat:.0f}, p = {p_mw:.6f}")
print("=" * 60)

# --- Save results to a text file for the repo ---
with open("results.txt", "w") as f:
    f.write(f"Mean difference: ${mean_diff:,.2f}\n")
    f.write(f"Welch t = {t_stat:.3f}, df = {df_welch:.1f}, p = {p_val:.6f}\n")
    f.write(f"95% CI = [${ci_low:,.2f}, ${ci_high:,.2f}]\n")
    f.write(f"Cohen's d = {cohens_d:.3f}\n")
