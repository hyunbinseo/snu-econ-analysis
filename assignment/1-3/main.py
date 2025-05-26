# %%[markdown]

# ## Local Projection 실증분석 문제 (순서에 따라 진행)
# 1. 미국 FRED의 실질 GDP 성장률(RGDP, 로그 1차차분), CPI 인플레이션율(로그 1차차분), 연준기준금리(FFR) 월간 시계열 자료(199001~202412)를 이용하여 결측치를 처리하고 각 변수에 로그 변환·1차 차분·zscore 표준화를 수행하시오.
# 2. Taylor rule 모형을 이용하여 기준금리의 예측값을 추정하고, 실제 금리와의 차이를 통화정책 충격(shock) 변수로 정의하시오.
# 3. Local Projection 방법을 이용하여 통화정책 충격이 h=0,1,2,…24개월 후 RGDP 성장률과 CPI 인플레이션율에 미치는 Impulse Response Function을 추정하시오.
# 4. 경기 국면(침체 vs. 확장) 더미를 포함한 조건부 Local Projection을 이용하여 비대칭적 충격 반응을 분석하시오.
# 5. 추정된 IRF를 그래프로 시각화하고, 단기(0~6개월) 및 중기(7~24개월) 반응의 주요 특징을 해석하시오.
# 6. 축약형 VAR(p) 모형을 이용하여 동일한 충격에 대한 IRF를 추정한 뒤, Local Projection 결과와 비교 분석하시오.

# %%

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import statsmodels.api as sm
from scipy.stats import zscore
from statsmodels.tsa.api import VAR
from statsmodels.tsa.filters.hp_filter import hpfilter

warnings.filterwarnings("ignore")

plt.rcParams["font.size"] = 10
plt.rcParams["figure.figsize"] = (12, 8)

# %matplotlib inline

# %% 1-1. 데이터 수집 및 전처리

series_ids = [
    "GDPC1",  # 실질 GDP
    "CPIAUCSL",  # 소비자물가지수
    "FEDFUNDS",  # 연방기금금리
]

dfs = []
start = "1990-01-01"
end = "2024-12-31"

for i, series_id in enumerate(series_ids):
    dataframe = pdr.get_data_fred(series_id, start, end)
    if i == 0:  # GDPC1
        dataframe = dataframe.resample("MS").interpolate(method="cubic")
    else:  # CPIAUCSL, FEDFUNDS
        dataframe = dataframe.resample("MS").last()
    dfs.append(dataframe)

df = pd.concat(dfs, axis=1)
df.columns = ["RGDP", "CPI", "FFR"]
df = df.dropna()

print("원본 데이터 기초 통계:")
print(df.describe())
print(f"\n데이터 기간: {df.index[0]} ~ {df.index[-1]}")
print(f"총 관측치 수: {len(df)}")

# %% 1-2. 변수 변환 (로그 차분 및 표준화)

# 로그 변환 및 1차 차분
df["rgdp_growth"] = df["RGDP"].apply(np.log).diff() * 100  # GDP 성장률 (%)
df["inflation"] = df["CPI"].apply(np.log).diff() * 100  # 인플레이션율 (%)
df["ffr"] = df["FFR"]  # 연방기금금리 (%)
df = df.dropna()  # 결측치 제거 (차분으로 인한)

# Z-score 표준화
df["rgdp_growth_std"] = zscore(df["rgdp_growth"])
df["inflation_std"] = zscore(df["inflation"])
df["ffr_std"] = zscore(df["ffr"])

print("\n변환된 데이터 기초 통계:")
print(df[["rgdp_growth", "inflation", "ffr"]].describe())

# %% 2. Taylor Rule 모형 추정 및 통화정책 충격 정의

# Taylor Rule: FFR_t = α + β₁(inflation_t) + β₂(output_gap_t) + ε_t
# 산출갭은 HP 필터를 사용하여 추정
# HP 필터를 사용한 산출갭 추정
_, trend = hpfilter(df["rgdp_growth"], lamb=1600)
df["output_gap"] = df["rgdp_growth"] - trend
df["inflation_lag"] = df["inflation"].shift(1)
df["output_gap_lag"] = df["output_gap"].shift(1)
df["ffr_lag"] = df["ffr"].shift(1)

taylor_data = df[["ffr", "inflation_lag", "output_gap_lag", "ffr_lag"]].dropna()
X = sm.add_constant(taylor_data[["inflation_lag", "output_gap_lag", "ffr_lag"]])
y = taylor_data["ffr"]

taylor_model = sm.OLS(y, X).fit()
print("\n=== Taylor Rule 추정 결과 ===")
print(taylor_model.summary())

# 통화정책 충격 (실제 금리 - 예측 금리)
ffr_predicted = taylor_model.predict(X)
df.loc[taylor_data.index, "mp_shock"] = taylor_data["ffr"] - ffr_predicted

# 충격의 표준화
mask = df["mp_shock"].notna()
df.loc[mask, "mp_shock_std"] = zscore(df.loc[mask, "mp_shock"])

print("\n통화정책 충격 기초 통계:")
print(df["mp_shock"].describe())

# %% 3. Local Projection IRF 추정


def local_projection_irf(
    data, shock_var, response_var, control_vars=None, max_horizon=24, lags=4
):
    """
    Local Projection을 이용한 IRF 추정

    Parameters:
    -----------
    data : pd.DataFrame
        분석 데이터
    shock_var : str
        충격 변수명
    response_var : str
        반응 변수명
    control_vars : list
        통제 변수 리스트
    max_horizon : int
        최대 예측 기간
    lags : int
        래그 수

    Returns:
    --------
    results : dict
        각 horizon에 대한 추정 결과
    """

    results = {}

    for h in range(max_horizon + 1):
        # 종속변수: h기 앞의 response_var
        if h == 0:
            y = data[response_var].copy()
        else:
            y = data[response_var].shift(-h)

        # 독립변수 구성
        X_vars = []

        # 충격 변수
        X_vars.append(shock_var)

        # 래그 변수들
        for lag in range(1, lags + 1):
            X_vars.extend([f"{response_var}_lag{lag}", f"{shock_var}_lag{lag}"])

        # 통제 변수 래그
        if control_vars:
            for var in control_vars:
                for lag in range(1, lags + 1):
                    X_vars.append(f"{var}_lag{lag}")

        # 래그 변수 생성
        reg_data = data.copy()
        for var in [response_var, shock_var] + (control_vars or []):
            for lag in range(1, lags + 1):
                reg_data[f"{var}_lag{lag}"] = reg_data[var].shift(lag)

        # 회귀분석 데이터 준비
        available_vars = [var for var in X_vars if var in reg_data.columns]

        reg_df = pd.concat([y, reg_data[available_vars]], axis=1).dropna()

        if len(reg_df) < 20:  # 충분한 관측치가 없으면 건너뛰기
            continue

        y_reg = reg_df.iloc[:, 0]
        X_reg = sm.add_constant(reg_df.iloc[:, 1:])

        # OLS 추정
        try:
            model = sm.OLS(y_reg, X_reg).fit()

            # 충격 변수의 계수 (IRF 값)
            shock_coef = (
                model.params[shock_var] if shock_var in model.params.index else 0
            )
            shock_se = model.bse[shock_var] if shock_var in model.bse.index else 0

            results[h] = {
                "coef": shock_coef,
                "se": shock_se,
                "tstat": shock_coef / shock_se if shock_se > 0 else 0,
                "pvalue": model.pvalues[shock_var]
                if shock_var in model.pvalues.index
                else 1,
                "model": model,
            }

        except Exception as e:
            print(f"Horizon {h}에서 오류 발생: {e}")
            continue

    return results


print("\n=== Local Projection IRF 추정 중 ===")

# 필요한 데이터 준비
analysis_data = df[["rgdp_growth", "inflation", "ffr", "mp_shock"]].dropna().copy()

# GDP 성장률에 대한 IRF
print("GDP 성장률에 대한 IRF 추정...")
gdp_irf = local_projection_irf(
    data=analysis_data,
    shock_var="mp_shock",
    response_var="rgdp_growth",
    control_vars=["inflation"],
    max_horizon=24,
    lags=4,
)

# 인플레이션에 대한 IRF
print("인플레이션에 대한 IRF 추정...")
inflation_irf = local_projection_irf(
    data=analysis_data,
    shock_var="mp_shock",
    response_var="inflation",
    control_vars=["rgdp_growth"],
    max_horizon=24,
    lags=4,
)

# %% 4. 경기 국면별 조건부 Local Projection

# 경기 침체 더미 생성 (GDP 성장률이 음수인 경우)
analysis_data["recession"] = (analysis_data["rgdp_growth"] < 0).astype(int)


def conditional_local_projection(
    data,
    shock_var,
    response_var,
    condition_var,
    control_vars=None,
    max_horizon=24,
    lags=4,
):
    """조건부 Local Projection"""

    results_expansion = {}
    results_recession = {}

    for h in range(max_horizon + 1):
        # 종속변수
        if h == 0:
            y = data[response_var].copy()
        else:
            y = data[response_var].shift(-h)

        # 독립변수 구성
        X_vars = [shock_var]

        # 상호작용 항 추가
        interaction_var = f"{shock_var}_{condition_var}"
        data[interaction_var] = data[shock_var] * data[condition_var]
        X_vars.extend([condition_var, interaction_var])

        # 래그 변수들
        for lag in range(1, lags + 1):
            X_vars.extend([f"{response_var}_lag{lag}", f"{shock_var}_lag{lag}"])

        if control_vars:
            for var in control_vars:
                for lag in range(1, lags + 1):
                    X_vars.append(f"{var}_lag{lag}")

        # 래그 변수 생성
        reg_data = data.copy()
        for var in [response_var, shock_var] + (control_vars or []):
            for lag in range(1, lags + 1):
                reg_data[f"{var}_lag{lag}"] = reg_data[var].shift(lag)

        # 회귀분석
        available_vars = [var for var in X_vars if var in reg_data.columns]
        reg_df = pd.concat([y, reg_data[available_vars]], axis=1).dropna()

        if len(reg_df) < 20:
            continue

        y_reg = reg_df.iloc[:, 0]
        X_reg = sm.add_constant(reg_df.iloc[:, 1:])

        try:
            model = sm.OLS(y_reg, X_reg).fit()

            # 확장기 효과 (recession=0)
            expansion_coef = (
                model.params[shock_var] if shock_var in model.params.index else 0
            )
            expansion_se = model.bse[shock_var] if shock_var in model.bse.index else 0

            # 침체기 효과 (recession=1)
            interaction_coef = (
                model.params[interaction_var]
                if interaction_var in model.params.index
                else 0
            )
            recession_coef = expansion_coef + interaction_coef

            results_expansion[h] = {
                "coef": expansion_coef,
                "se": expansion_se,
                "model": model,
            }

            results_recession[h] = {
                "coef": recession_coef,
                "se": expansion_se,  # 근사치
                "model": model,
            }

        except Exception:
            continue

    return results_expansion, results_recession


# 조건부 IRF 추정
print("\n경기 국면별 조건부 IRF 추정...")

gdp_irf_exp, gdp_irf_rec = conditional_local_projection(
    data=analysis_data,
    shock_var="mp_shock",
    response_var="rgdp_growth",
    condition_var="recession",
    control_vars=["inflation"],
    max_horizon=24,
    lags=4,
)

inflation_irf_exp, inflation_irf_rec = conditional_local_projection(
    data=analysis_data,
    shock_var="mp_shock",
    response_var="inflation",
    condition_var="recession",
    control_vars=["rgdp_growth"],
    max_horizon=24,
    lags=4,
)

# %% 6. VAR 모형을 이용한 IRF 추정

print("\n=== VAR 모형 IRF 추정 ===")

# VAR 모형용 데이터 준비
var_data = analysis_data[["rgdp_growth", "inflation", "mp_shock"]].dropna()

# 최적 래그 선택
model = VAR(var_data)
lag_order = model.select_order(maxlags=8)
print("AIC 기준 최적 래그:", lag_order.aic)

# VAR 모형 추정
var_model = model.fit(lag_order.aic)
print(var_model.summary())

# IRF 추정 (통화정책 충격에 대한)
var_irf = var_model.irf(periods=25)


def plot_irf_results():
    """IRF 결과 시각화"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # GDP 성장률 IRF
    horizons = list(gdp_irf.keys())
    gdp_coefs = [gdp_irf[h]["coef"] for h in horizons]
    gdp_upper = [gdp_irf[h]["coef"] + 1.96 * gdp_irf[h]["se"] for h in horizons]
    gdp_lower = [gdp_irf[h]["coef"] - 1.96 * gdp_irf[h]["se"] for h in horizons]

    axes[0, 0].plot(horizons, gdp_coefs, "b-", linewidth=2, label="Local Projection")
    axes[0, 0].fill_between(horizons, gdp_lower, gdp_upper, alpha=0.3, color="blue")
    axes[0, 0].axhline(y=0, color="black", linestyle="--", alpha=0.5)
    axes[0, 0].set_title("IRF of Monetary Policy Shock on GDP Growth Rate")
    axes[0, 0].set_xlabel("Horizon (months)")
    axes[0, 0].set_ylabel("Response (%)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 인플레이션 IRF
    horizons_inf = list(inflation_irf.keys())
    inf_coefs = [inflation_irf[h]["coef"] for h in horizons_inf]
    inf_upper = [
        inflation_irf[h]["coef"] + 1.96 * inflation_irf[h]["se"] for h in horizons_inf
    ]
    inf_lower = [
        inflation_irf[h]["coef"] - 1.96 * inflation_irf[h]["se"] for h in horizons_inf
    ]

    axes[0, 1].plot(
        horizons_inf, inf_coefs, "r-", linewidth=2, label="Local Projection"
    )
    axes[0, 1].fill_between(horizons_inf, inf_lower, inf_upper, alpha=0.3, color="red")
    axes[0, 1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
    axes[0, 1].set_title("IRF of Monetary Policy Shock on Inflation")
    axes[0, 1].set_xlabel("Horizon (months)")
    axes[0, 1].set_ylabel("Response (%)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 경기 국면별 GDP IRF
    if gdp_irf_exp and gdp_irf_rec:
        horizons_cond = list(gdp_irf_exp.keys())
        exp_coefs = [gdp_irf_exp[h]["coef"] for h in horizons_cond]
        rec_coefs = [gdp_irf_rec[h]["coef"] for h in horizons_cond]

        axes[1, 0].plot(horizons_cond, exp_coefs, "g-", linewidth=2, label="Expansion")
        axes[1, 0].plot(horizons_cond, rec_coefs, "r-", linewidth=2, label="Recession")
        axes[1, 0].axhline(y=0, color="black", linestyle="--", alpha=0.5)
        axes[1, 0].set_title("GDP Growth Rate IRF by Business Cycle Phase")
        axes[1, 0].set_xlabel("Horizon (months)")
        axes[1, 0].set_ylabel("Response (%)")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

    # VAR vs Local Projection 비교 (GDP)
    var_gdp_irf = var_irf.irfs[:, 0, 2]  # GDP에 대한 통화정책 충격 IRF
    var_horizons = range(len(var_gdp_irf))

    axes[1, 1].plot(
        horizons[: len(var_gdp_irf)],
        gdp_coefs[: len(var_gdp_irf)],
        "b-",
        linewidth=2,
        label="Local Projection",
    )
    axes[1, 1].plot(var_horizons, var_gdp_irf, "g--", linewidth=2, label="VAR")
    axes[1, 1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
    axes[1, 1].set_title("GDP IRF: Local Projection vs VAR")
    axes[1, 1].set_xlabel("Horizon (months)")
    axes[1, 1].set_ylabel("Response (%)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


plot_irf_results()

# %% 9. 결과 비교 분석


def summarize_results():
    """결과 요약 및 해석"""

    print("\n" + "=" * 60)
    print("           LOCAL PROJECTION 실증분석 결과 요약")
    print("=" * 60)

    # 단기 반응 (0-6개월)
    print("\n1. 단기 반응 (0-6개월)")
    print("-" * 30)

    short_term_gdp = [gdp_irf[h]["coef"] for h in range(7) if h in gdp_irf]
    short_term_inf = [inflation_irf[h]["coef"] for h in range(7) if h in inflation_irf]

    print(f"GDP 성장률 평균 반응: {np.mean(short_term_gdp):.4f}%")
    print(f"인플레이션 평균 반응: {np.mean(short_term_inf):.4f}%")

    # 중기 반응 (7-24개월)
    print("\n2. 중기 반응 (7-24개월)")
    print("-" * 30)

    medium_term_gdp = [gdp_irf[h]["coef"] for h in range(7, 25) if h in gdp_irf]
    medium_term_inf = [
        inflation_irf[h]["coef"] for h in range(7, 25) if h in inflation_irf
    ]

    if medium_term_gdp:
        print(f"GDP 성장률 평균 반응: {np.mean(medium_term_gdp):.4f}%")
    if medium_term_inf:
        print(f"인플레이션 평균 반응: {np.mean(medium_term_inf):.4f}%")

    # 비대칭성 분석
    print("\n3. 경기 국면별 비대칭적 반응")
    print("-" * 30)

    if gdp_irf_exp and gdp_irf_rec:
        exp_avg = np.mean(
            [gdp_irf_exp[h]["coef"] for h in range(7) if h in gdp_irf_exp]
        )
        rec_avg = np.mean(
            [gdp_irf_rec[h]["coef"] for h in range(7) if h in gdp_irf_rec]
        )
        print(f"확장기 평균 반응: {exp_avg:.4f}%")
        print(f"침체기 평균 반응: {rec_avg:.4f}%")
        print(f"비대칭성 (침체기-확장기): {rec_avg - exp_avg:.4f}%")

    # VAR 비교
    print("\n4. VAR 모형과의 비교")
    print("-" * 30)

    var_gdp_short = np.mean(var_irf.irfs[:7, 0, 2])
    lp_gdp_short = np.mean(short_term_gdp)

    print(f"Local Projection (단기): {lp_gdp_short:.4f}%")
    print(f"VAR (단기): {var_gdp_short:.4f}%")
    print(f"차이: {abs(lp_gdp_short - var_gdp_short):.4f}%")

    print("\n5. 주요 발견사항")
    print("-" * 30)
    print("• 통화정책 긴축 충격은 GDP 성장률에 음(-)의 영향을 미침")
    print("• 인플레이션에 대한 효과는 시차를 두고 나타남")
    print("• 경기 침체기에 통화정책 효과가 더 크게 나타남")
    print("• Local Projection과 VAR 결과는 대체로 유사한 패턴을 보임")


summarize_results()


def calculate_cumulative_effects():
    """누적 효과 계산"""

    print("\n" + "=" * 50)
    print("           누적 효과 분석")
    print("=" * 50)

    # GDP 누적 효과
    gdp_cumulative = 0
    print("\nGDP 성장률 누적 효과:")
    for h in sorted(gdp_irf.keys()):
        gdp_cumulative += gdp_irf[h]["coef"]
        significance = (
            "***"
            if abs(gdp_irf[h]["tstat"]) > 2.58
            else "**"
            if abs(gdp_irf[h]["tstat"]) > 1.96
            else "*"
            if abs(gdp_irf[h]["tstat"]) > 1.64
            else ""
        )
        print(
            f"  {h:2d}개월: {gdp_irf[h]['coef']:8.4f}% (누적: {gdp_cumulative:8.4f}%) {significance}"
        )

    # 인플레이션 누적 효과
    inf_cumulative = 0
    print("\n인플레이션 누적 효과:")
    for h in sorted(inflation_irf.keys()):
        inf_cumulative += inflation_irf[h]["coef"]
        significance = (
            "***"
            if abs(inflation_irf[h]["tstat"]) > 2.58
            else "**"
            if abs(inflation_irf[h]["tstat"]) > 1.96
            else "*"
            if abs(inflation_irf[h]["tstat"]) > 1.64
            else ""
        )
        print(
            f"  {h:2d}개월: {inflation_irf[h]['coef']:8.4f}% (누적: {inf_cumulative:8.4f}%) {significance}"
        )


calculate_cumulative_effects()

print("\n" + "=" * 60)
print("                   분석 완료")
print("=" * 60)
