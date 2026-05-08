import re
import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs


# =========================================================
# 設定
# =========================================================
K1_MIN = 20
ODDS_MIN = 100.0
RANK_MAX = 40


# =========================================================
# 確率・本気帯検出
# =========================================================
def prob_power_from_odds(odds, alpha=1.0):
    odds = np.asarray(odds, dtype=float)

    if np.any(~np.isfinite(odds)) or np.any(odds <= 0):
        raise ValueError("oddsに不正値があります。")

    w = (1.0 / odds) ** alpha
    s = w.sum()

    if not np.isfinite(s) or s <= 0:
        raise ValueError("確率正規化に失敗しました。")

    return w / s


def detect_serious_band_by_power_cross(
    race_df: pd.DataFrame,
    min_serious_len=8,
    alpha_low=0.85,
    alpha_high=1.30,
):
    if "odds" not in race_df.columns:
        raise ValueError("race_df に odds 列が必要です。")

    odds = race_df["odds"].to_numpy(dtype=float)
    n = len(race_df)

    if n < min_serious_len + 1:
        raise ValueError("点数が少なすぎます。")

    p_low = prob_power_from_odds(odds, alpha=alpha_low)
    p_high = prob_power_from_odds(odds, alpha=alpha_high)
    diff = p_high - p_low

    neg_idx = np.where(diff < 0)[0]

    if len(neg_idx) == 0:
        k1 = n
        cross_rank_float = float(n)
        cross_found = False
    else:
        first_neg = int(neg_idx[0])
        k1 = first_neg

        if first_neg == 0:
            cross_rank_float = 1.0
        else:
            d0 = diff[first_neg - 1]
            d1 = diff[first_neg]
            frac = 0.0 if d0 == d1 else d0 / (d0 - d1)
            cross_rank_float = first_neg + frac

        cross_found = True

    k1 = max(k1, min_serious_len)
    k1 = min(k1, n)

    return {
        "k1": int(k1),
        "cross_found": cross_found,
        "cross_rank_float": float(cross_rank_float),
    }


# =========================================================
# BOATRACE URL処理
# =========================================================
def parse_boatrace_url(url: str) -> dict:
    q = parse_qs(urlparse(url).query)

    required = ["hd", "jcd", "rno"]
    missing = [x for x in required if x not in q]
    if missing:
        raise ValueError(f"URLに必要なパラメータがありません: {missing}")

    return {
        "hd": q["hd"][0],
        "jcd": q["jcd"][0].zfill(2),
        "rno": str(int(q["rno"][0])),
    }


def build_official_url(params: dict) -> str:
    return (
        "https://www.boatrace.jp/owpc/pc/race/odds3t"
        f"?rno={params['rno']}&jcd={params['jcd']}&hd={params['hd']}"
    )


def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    res = requests.get(url, headers=headers, timeout=15)
    res.raise_for_status()
    res.encoding = res.apparent_encoding
    return res.text


def extract_odds_lines_from_html(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")

    lines = [
        line.strip()
        for line in soup.get_text("\n").splitlines()
        if line.strip()
    ]

    start_idx = None
    for i, line in enumerate(lines):
        if "3連単オッズ" in line:
            start_idx = i
            break

    if start_idx is None:
        raise ValueError("3連単オッズの見出しが見つかりません。")

    odds_lines = []
    for line in lines[start_idx + 1:]:
        if "ボートレースガイド" in line or "PAGE TOP" in line:
            break

        if re.fullmatch(r"[0-9.\s]+", line):
            nums = line.split()

            # 1行 = first 1〜6列ぶんの second, third, odds の18トークン想定
            if len(nums) == 18:
                odds_lines.append(line)

    if len(odds_lines) != 20:
        raise ValueError(f"オッズ表20行が取れませんでした。取得行数={len(odds_lines)}")

    return odds_lines


def parse_odds3t_table_from_lines(odds_lines: list[str]) -> pd.DataFrame:
    rows = []

    for line in odds_lines:
        nums = line.split()

        if len(nums) != 18:
            raise ValueError(f"想定外の行です: {line}")

        for first_idx in range(6):
            first = first_idx + 1
            base = first_idx * 3

            second = int(nums[base])
            third = int(nums[base + 1])
            odds = float(nums[base + 2])

            rows.append({
                "first": first,
                "second": second,
                "third": third,
                "odds": odds,
            })

    df = pd.DataFrame(rows)

    df["label"] = (
        df["first"].astype(str) + "-"
        + df["second"].astype(str) + "-"
        + df["third"].astype(str)
    )

    if len(df) != 120:
        raise ValueError(f"3連単が120点ではありません: {len(df)}")

    if df["label"].duplicated().any():
        dup = df[df["label"].duplicated(keep=False)]
        raise ValueError(f"重複ラベルがあります:\n{dup}")

    invalid = df[
        (df["first"] == df["second"]) |
        (df["first"] == df["third"]) |
        (df["second"] == df["third"])
    ]
    if len(invalid) > 0:
        raise ValueError(f"不正な組み合わせがあります:\n{invalid}")

    df = df.sort_values("odds", ascending=True).reset_index(drop=True)
    df["rank"] = df.index + 1

    return df


def fetch_odds3t_from_url(url: str) -> pd.DataFrame:
    params = parse_boatrace_url(url)
    official_url = build_official_url(params)

    html = fetch_html(official_url)
    odds_lines = extract_odds_lines_from_html(html)
    df = parse_odds3t_table_from_lines(odds_lines)

    df["hd"] = params["hd"]
    df["jcd"] = params["jcd"]
    df["rno"] = params["rno"]

    return df


def pick_bets_from_url(url: str) -> dict:
    race_df = fetch_odds3t_from_url(url)

    band = detect_serious_band_by_power_cross(race_df)
    k1 = band["k1"]

    if k1 < K1_MIN:
        picks = race_df.iloc[0:0].copy()
    else:
        picks = race_df[
            (race_df["odds"] >= ODDS_MIN) &
            (race_df["rank"] <= RANK_MAX)
        ].copy()

    return {
        "k1": k1,
        "cross_found": band["cross_found"],
        "cross_rank_float": band["cross_rank_float"],
        "race_df": race_df,
        "picks": picks,
    }


# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(
    page_title="BOATRACE 3連単 買い目判定",
    layout="wide",
)

st.title("BOATRACE 3連単 買い目判定")

st.caption(
    "条件: k1 >= 20 / odds >= 100 / rank <= 40"
)

default_url = "https://www.boatrace.jp/owpc/pc/race/odds3t?rno=7&jcd=04&hd=20260508"

url = st.text_input(
    "BOATRACE公式の3連単オッズURLを貼ってください",
    value=default_url,
)

if st.button("判定"):
    try:
        params = parse_boatrace_url(url)
        result = pick_bets_from_url(url)

        k1 = result["k1"]
        picks = result["picks"]
        race_df = result["race_df"]

        st.subheader("判定結果")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("日付", params["hd"])
        col2.metric("場コード", params["jcd"])
        col3.metric("R", params["rno"])
        col4.metric("k1", k1)

        st.write(f"cross_found: `{result['cross_found']}`")
        st.write(f"cross_rank_float: `{result['cross_rank_float']:.2f}`")

        if k1 < K1_MIN:
            st.warning(f"見送り: k1={k1} < {K1_MIN}")
        elif len(picks) == 0:
            st.warning("買い目なし: 条件に一致する買い目がありません。")
        else:
            st.success(f"買い目候補: {len(picks)}点")

            display_picks = picks[["label", "odds", "rank", "first", "second", "third"]].copy()
            display_picks = display_picks.sort_values("rank")

            st.dataframe(display_picks, use_container_width=True)

            text_lines = [
                f"{row.label}  {row.odds:.1f}倍  rank={int(row.rank)}"
                for row in display_picks.itertuples()
            ]
            st.text_area(
                "コピー用",
                value="\n".join(text_lines),
                height=120,
            )

        with st.expander("全120点オッズを見る"):
            st.dataframe(
                race_df[["label", "odds", "rank", "first", "second", "third"]],
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"エラー: {e}")
