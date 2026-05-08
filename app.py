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
# URL処理
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


# =========================================================
# 3連単オッズパーサー
# =========================================================
def is_numeric_token(token: str) -> bool:
    """
    数値セル判定。
    例:
      20.4
      1086
      1,086
    """
    token = token.replace(",", "")
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", token))


def to_number(token: str) -> float:
    return float(token.replace(",", ""))


def extract_odds_tokens_from_html(html: str) -> list[str]:
    """
    BOATRACE公式の3連単オッズ表を、数字トークン列として抽出する。

    実際のHTMLテキストは、表の各セルが改行で分かれているため、
    行単位ではなくトークン単位で読む。

    3連単オッズ表の数字トークンは以下:
      選手ヘッダ: 1,2,3,4,5,6  → 除外
      オッズ表本体: 270 tokens
    """
    soup = BeautifulSoup(html, "html.parser")

    tokens = [
        x.replace("\xa0", " ").strip()
        for x in soup.get_text("\n").splitlines()
        if x.replace("\xa0", " ").strip()
    ]

    start_idx = None
    for i, token in enumerate(tokens):
        if "3連単オッズ" in token:
            start_idx = i
            break

    if start_idx is None:
        raise ValueError("3連単オッズの見出しが見つかりません。")

    numeric_tokens = []

    for token in tokens[start_idx + 1:]:
        # 表の終端
        if (
            "締切時オッズは" in token
            or "ボートレースガイド" in token
            or token == "投票"
            or token == "PAGE TOP"
        ):
            break

        if is_numeric_token(token):
            numeric_tokens.append(token.replace(",", ""))

    # 3連単オッズ直後の選手ヘッダ番号 1,2,3,4,5,6 を除外
    if len(numeric_tokens) >= 6 and numeric_tokens[:6] == ["1", "2", "3", "4", "5", "6"]:
        numeric_tokens = numeric_tokens[6:]

    if len(numeric_tokens) != 270:
        preview = numeric_tokens[:120]
        raise ValueError(
            "3連単オッズ用の数字トークン数が想定外です。\n"
            f"取得数={len(numeric_tokens)} / 想定=270\n"
            f"先頭プレビュー={preview}"
        )

    return numeric_tokens


def parse_odds3t_table_from_tokens(tokens: list[str]) -> pd.DataFrame:
    """
    3連単オッズの数字トークン列270個から120点を復元する。

    表構造:
      5つのsecondグループ
      各グループは4行
      各行はfirst=1..6の6列

    各secondグループの先頭行:
      second, third, odds × 6列 = 18 tokens

    継続3行:
      third, odds × 6列 = 12 tokens

    1グループ = 18 + 12*3 = 54 tokens
    5グループ = 270 tokens
    """
    if len(tokens) != 270:
        raise ValueError(f"tokens は270個必要です。現在: {len(tokens)}")

    rows = []
    idx = 0
    current_second_by_first = {}

    for second_group in range(5):
        for row_in_group in range(4):
            for first in range(1, 7):
                if row_in_group == 0:
                    second = int(to_number(tokens[idx]))
                    third = int(to_number(tokens[idx + 1]))
                    odds = float(to_number(tokens[idx + 2]))
                    idx += 3

                    current_second_by_first[first] = second

                    rows.append({
                        "first": first,
                        "second": second,
                        "third": third,
                        "odds": odds,
                    })
                else:
                    if first not in current_second_by_first:
                        raise ValueError(f"first={first} の second が未設定です。")

                    second = current_second_by_first[first]
                    third = int(to_number(tokens[idx]))
                    odds = float(to_number(tokens[idx + 1]))
                    idx += 2

                    rows.append({
                        "first": first,
                        "second": second,
                        "third": third,
                        "odds": odds,
                    })

    if idx != len(tokens):
        raise ValueError(f"tokensを最後まで消費できていません。idx={idx}, len={len(tokens)}")

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
    tokens = extract_odds_tokens_from_html(html)
    df = parse_odds3t_table_from_tokens(tokens)

    df["hd"] = params["hd"]
    df["jcd"] = params["jcd"]
    df["rno"] = params["rno"]

    return df


# =========================================================
# 買い目判定
# =========================================================
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
    f"条件: k1 >= {K1_MIN} / odds >= {ODDS_MIN:.0f} / rank <= {RANK_MAX}"
)

default_url = "https://www.boatrace.jp/owpc/pc/race/odds3t?rno=8&jcd=04&hd=20260508"

url = st.text_input(
    "BOATRACE公式の3連単オッズURLを貼ってください",
    value=default_url,
)

debug = st.checkbox("デバッグ情報を表示する", value=False)

if st.button("判定"):
    try:
        params = parse_boatrace_url(url)
        official_url = build_official_url(params)

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

        st.write(f"公式URL: {official_url}")
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

        if debug:
            with st.expander("デバッグ: 取得トークン確認"):
                html = fetch_html(official_url)
                tokens = extract_odds_tokens_from_html(html)
                st.write(f"tokens count: {len(tokens)}")
                st.write(tokens[:120])

    except Exception as e:
        st.error(f"エラー: {e}")

        if debug:
            st.exception(e)
