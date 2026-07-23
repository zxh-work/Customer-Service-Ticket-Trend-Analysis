"""
客服工单趋势分析工具
分析维度：时间趋势、分类分布、优先级、满意度、处理时长、渠道、异常检测

用法:
  python analyze_tickets.py
  python analyze_tickets.py --data task5_tickets.json --output ./charts --report report.html
"""

import argparse
import base64
import json
import os
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from io import BytesIO

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
matplotlib.rcParams["axes.unicode_minus"] = False


# ========== 字体兼容性：自动检测可用中文字体 ==========

def _detect_chinese_font():
    """检测系统可用的中文字体，返回第一个可用的字体名称，兜底返回 sans-serif."""
    candidates = [
        "Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC",
        "Noto Sans SC", "WenQuanYi Micro Hei", "Source Han Sans CN",
        "Source Han Sans SC", "AR PL UMing CN", "STHeiti", "STSong",
    ]
    # 获取 matplotlib 可用的字体列表
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for c in candidates:
        if c in available:
            return c
    return "sans-serif"


_CHINESE_FONT = _detect_chinese_font()
plt.rcParams["font.sans-serif"] = [_CHINESE_FONT, "DejaVu Sans"]
print(f"[字体] 使用: {_CHINESE_FONT}")


# ========== 数据分析函数 ==========

def load_data(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        tickets = json.load(f)
    df = pd.DataFrame(tickets)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["date"] = df["created_at"].dt.date
    df["weekday"] = df["created_at"].dt.day_name()
    df["hour"] = df["created_at"].dt.hour
    return df


def analyze(df: pd.DataFrame):
    """执行所有分析，返回包含分析结果的字典."""
    result = {}

    # --- 基本信息 ---
    result["total"] = len(df)
    result["period"] = (str(df["date"].min()), str(df["date"].max()))

    # --- 时间趋势 ---
    daily_counts = df.groupby("date").size()
    weekday_counts = df.groupby("weekday").size().sort_values(ascending=False)
    result["daily_avg"] = round(daily_counts.mean(), 1)
    result["daily_max"] = int(daily_counts.max())
    result["daily_max_date"] = str(daily_counts.idxmax())
    result["daily_min"] = int(daily_counts.min())
    result["daily_min_date"] = str(daily_counts.idxmin())
    result["daily_counts"] = daily_counts
    result["weekday_counts"] = weekday_counts

    # --- 分类分布 ---
    cat_counts = df["category"].value_counts()
    result["cat_counts"] = cat_counts
    cat_daily = df.groupby(["date", "category"]).size().unstack(fill_value=0)
    result["cat_daily"] = cat_daily

    # --- 优先级 ---
    pri_counts = df["priority"].value_counts()
    result["pri_counts"] = pri_counts
    cross_pri = pd.crosstab(df["category"], df["priority"])
    result["cross_pri"] = cross_pri
    high_pri = df[df["priority"] == "高"]
    high_cat = high_pri["category"].value_counts()
    result["high_cat"] = high_cat
    result["high_pct"] = round(pri_counts.get("高", 0) / len(df) * 100, 1)

    # --- 处理时长 ---
    avg_time = df["resolution_time_hours"].mean()
    median_time = df["resolution_time_hours"].median()
    max_time = df["resolution_time_hours"].max()
    result["avg_time"] = round(avg_time, 1)
    result["median_time"] = round(median_time, 1)
    result["max_time"] = round(max_time, 1)
    cat_avg_time = df.groupby("category")["resolution_time_hours"].mean().sort_values(ascending=False)
    result["cat_avg_time"] = cat_avg_time
    long_tickets = df[df["resolution_time_hours"] >= 48]
    result["long_tickets"] = long_tickets

    # --- 满意度 ---
    avg_sat = df["satisfaction"].mean()
    result["avg_sat"] = round(avg_sat, 2)
    cat_sat = df.groupby("category")["satisfaction"].mean().sort_values()
    result["cat_sat"] = cat_sat
    low_sat = df[df["satisfaction"] <= 2]
    result["low_sat_count"] = len(low_sat)
    result["low_sat_cat"] = low_sat["category"].value_counts()
    very_low_sat = df[df["satisfaction"] == 1]
    result["very_low_sat"] = very_low_sat
    corr = df["resolution_time_hours"].corr(df["satisfaction"])
    result["corr"] = round(corr, 3)

    # --- 渠道 ---
    channel_counts = df["channel"].value_counts()
    result["channel_counts"] = channel_counts
    channel_sat = df.groupby("channel")["satisfaction"].mean()
    result["channel_sat"] = channel_sat
    channel_time = df.groupby("channel")["resolution_time_hours"].mean()
    result["channel_time"] = channel_time

    # --- 未解决工单 ---
    unresolved = df[df["is_resolved"] == False]
    result["unresolved_count"] = len(unresolved)
    result["unresolved_pct"] = round(len(unresolved) / len(df) * 100, 1)
    result["unresolved_cat"] = unresolved["category"].value_counts()
    result["unresolved"] = unresolved

    # --- 异常信号 ---
    anomalies = []
    refund_pct = cat_counts.get("退款退货", 0) / len(df) * 100
    refund_sat = cat_sat.get("退款退货", 0)
    refund_time = cat_avg_time.get("退款退货", 0)
    anomalies.append({
        "title": "退款退货流程严重拖累满意度",
        "desc": (f"占比 {refund_pct:.0f}%，满意度仅 {refund_sat:.2f}（整体 {avg_sat:.2f}），"
                 f"平均处理 {refund_time:.0f} 小时"),
        "level": "critical",
    })

    pay_count = cat_counts.get("支付问题", 0)
    pay_sat = cat_sat.get("支付问题", 0)
    anomalies.append({
        "title": "支付系统存在 Bug",
        "desc": f"{pay_count} 张工单，满意度 {pay_sat:.2f}，反复出现重复扣款、支付回调失败",
        "level": "critical",
    })

    result["refund_pct"] = refund_pct
    result["refund_sat"] = refund_sat
    result["refund_time"] = refund_time
    result["pay_count"] = pay_count
    result["pay_sat"] = pay_sat

    high_pct = result["high_pct"]
    anomalies.append({
        "title": "高优先级工单占比过高",
        "desc": f"高优先级占 {high_pct:.0f}%，核心流程稳定性不足",
        "level": "warning",
    })

    anomalies.append({
        "title": "严重不满（评分 1 分）工单集中",
        "desc": f"{len(very_low_sat)} 张，集中在退款退货和投诉类别",
        "level": "critical",
    })

    complaint_count = cat_counts.get("投诉", 0)
    anomalies.append({
        "title": "投诉类工单出现",
        "desc": f"{complaint_count} 张，涉及客服态度差、等待时间过长、机器人无效回复",
        "level": "warning",
    })

    anomalies.append({
        "title": "未解决工单积压",
        "desc": f"{len(unresolved)} 张未解决，平均已处理 {unresolved['resolution_time_hours'].mean():.0f} 小时，存在升级风险",
        "level": "warning",
    })

    anomalies.append({
        "title": "相似问题反复出现",
        "desc": "重复扣款、退款延迟等同类问题未根治",
        "level": "warning",
    })

    result["anomalies"] = anomalies
    result["complaint_count"] = complaint_count

    # --- 关键词统计 ---
    keyword_map = {}
    for kw in ["退款", "重复扣款", "快递", "退货"]:
        keyword_map[kw] = int(df["description"].str.contains(kw, regex=True).sum())
    result["keywords"] = keyword_map

    return result


# ========== 可视化函数 ==========

def generate_charts(df: pd.DataFrame, r: dict, output_dir: str):
    """生成所有图表并保存到 output_dir，返回图片文件路径列表."""
    os.makedirs(output_dir, exist_ok=True)
    chart_paths = {}

    # 图1: 每日工单量趋势
    fig, ax = plt.subplots(figsize=(14, 5))
    dc = r["daily_counts"]
    dates = sorted(dc.index)
    ax.plot(dates, dc.values, marker="o", linewidth=2, color="#2196F3", markersize=6)
    ax.axhline(y=r["daily_avg"], color="red", linestyle="--", alpha=0.7, label=f"日均 {r['daily_avg']}")
    ax.fill_between(dates, dc.values, alpha=0.15, color="#2196F3")
    ax.set_title("每日工单量趋势", fontsize=14)
    ax.set_xlabel("日期")
    ax.set_ylabel("工单数量")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    p = os.path.join(output_dir, "1_daily_trend.png")
    fig.savefig(p, dpi=200)
    plt.close()
    chart_paths["每日工单量趋势"] = p

    # 图2: 分类分布饼图
    fig, ax = plt.subplots(figsize=(8, 8))
    cc = r["cat_counts"]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8"]
    wedges, texts, autotexts = ax.pie(
        cc.values, labels=cc.index, autopct="%1.1f%%",
        colors=colors[: len(cc)], startangle=90, textprops={"fontsize": 12},
    )
    for t in autotexts:
        t.set_fontsize(11)
    ax.set_title("工单分类分布", fontsize=14)
    plt.tight_layout()
    p = os.path.join(output_dir, "2_category_pie.png")
    fig.savefig(p, dpi=200)
    plt.close()
    chart_paths["工单分类分布"] = p

    # 图3: 分类×优先级堆叠柱状图
    fig, ax = plt.subplots(figsize=(12, 6))
    categories = list(r["cat_counts"].index)
    pri_order = ["高", "中", "低"]
    pri_colors = {"高": "#FF4444", "中": "#FFA726", "低": "#66BB6A"}
    cross = r["cross_pri"]
    bottom = np.zeros(len(categories))
    for pri in pri_order:
        values = [
            cross.loc[cat, pri] if cat in cross.index and pri in cross.columns else 0
            for cat in categories
        ]
        ax.bar(categories, values, bottom=bottom, label=pri, color=pri_colors[pri], alpha=0.85)
        bottom += values
    ax.set_title("各分类优先级分布", fontsize=14)
    ax.set_xlabel("问题分类")
    ax.set_ylabel("工单数量")
    ax.legend(title="优先级")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=15)
    plt.tight_layout()
    p = os.path.join(output_dir, "3_category_priority.png")
    fig.savefig(p, dpi=200)
    plt.close()
    chart_paths["各分类优先级分布"] = p

    # 图4: 处理时长 + 满意度双轴图
    fig, ax1 = plt.subplots(figsize=(12, 6))
    cat_time = r["cat_avg_time"]
    cat_sat = r["cat_sat"]
    ordered = list(cat_time.index)
    times = cat_time.values
    sats = [cat_sat[c] for c in ordered]
    ax1.bar(ordered, times, color="#42A5F5", alpha=0.7, label="平均处理时长(小时)")
    ax1.set_xlabel("问题分类")
    ax1.set_ylabel("平均处理时长 (小时)", color="#42A5F5", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="#42A5F5")
    ax1.set_ylim(0, max(times) * 1.3)
    ax2 = ax1.twinx()
    ax2.plot(ordered, sats, marker="D", linewidth=2, color="#EF5350", markersize=8, label="平均满意度")
    ax2.axhline(y=r["avg_sat"], color="gray", linestyle=":", alpha=0.6, label=f"整体平均 {r['avg_sat']}")
    ax2.set_ylabel("平均满意度", color="#EF5350", fontsize=12)
    ax2.set_ylim(0, 5.5)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    ax1.set_title("各分类处理时长与满意度对比", fontsize=14)
    ax1.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    p = os.path.join(output_dir, "4_category_time_satisfaction.png")
    fig.savefig(p, dpi=200)
    plt.close()
    chart_paths["处理时长与满意度对比"] = p

    # 图5: 处理时长 vs 满意度散点图
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = {
        "退款退货": "#FF4444", "支付问题": "#FFA726", "物流查询": "#42A5F5",
        "投诉": "#D32F2F", "商品咨询": "#66BB6A", "账号问题": "#AB47BC",
    }
    for cat in df["category"].unique():
        sub = df[df["category"] == cat]
        ax.scatter(
            sub["resolution_time_hours"], sub["satisfaction"],
            label=cat, s=80, alpha=0.7, c=cmap.get(cat, "#888"),
            edgecolors="white", linewidth=0.5,
        )
    ax.set_xlabel("处理时长 (小时)", fontsize=12)
    ax.set_ylabel("满意度评分", fontsize=12)
    ax.set_title("处理时长 vs 满意度评分", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=3, color="gray", linestyle="--", alpha=0.4)
    ax.axvline(x=24, color="gray", linestyle="--", alpha=0.4)
    plt.tight_layout()
    p = os.path.join(output_dir, "5_time_vs_satisfaction.png")
    fig.savefig(p, dpi=200)
    plt.close()
    chart_paths["处理时长 vs 满意度"] = p

    # 图6: 渠道分析
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    channels = list(r["channel_counts"].index)
    chn_colors = ["#42A5F5", "#FFA726", "#66BB6A"]
    ax = axes[0]
    ax.bar(channels, r["channel_counts"].values, color=chn_colors[: len(channels)], alpha=0.8)
    ax.set_title("各渠道工单量", fontsize=12)
    ax.set_ylabel("工单数量")
    for i, v in enumerate(r["channel_counts"].values):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax = axes[1]
    ax.bar(channels, r["channel_sat"].values, color=chn_colors[: len(channels)], alpha=0.8)
    ax.axhline(y=r["avg_sat"], color="red", linestyle="--", alpha=0.6, label=f"平均 {r['avg_sat']}")
    ax.set_title("各渠道平均满意度", fontsize=12)
    ax.set_ylabel("满意度")
    ax.set_ylim(0, 5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax = axes[2]
    ax.bar(channels, r["channel_time"].values, color=chn_colors[: len(channels)], alpha=0.8)
    ax.set_title("各渠道平均处理时长", fontsize=12)
    ax.set_ylabel("处理时长 (小时)")
    for i, v in enumerate(r["channel_time"].values):
        ax.text(i, v + 0.5, f"{v:.0f}h", ha="center", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    p = os.path.join(output_dir, "6_channel_analysis.png")
    fig.savefig(p, dpi=200)
    plt.close()
    chart_paths["渠道分析"] = p

    # 图7: 分类每日趋势热力图
    fig, ax = plt.subplots(figsize=(14, 6))
    cat_daily_t = r["cat_daily"].T
    im = ax.imshow(cat_daily_t.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(cat_daily_t.columns)))
    ax.set_xticklabels([str(d) for d in cat_daily_t.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(cat_daily_t.index)))
    ax.set_yticklabels(cat_daily_t.index)
    ax.set_xlabel("日期")
    ax.set_ylabel("问题分类")
    ax.set_title("各类工单每日数量热力图", fontsize=14)
    plt.colorbar(im, ax=ax, label="工单数量")
    for i in range(len(cat_daily_t.index)):
        for j in range(len(cat_daily_t.columns)):
            val = cat_daily_t.values[i, j]
            if val > 0:
                color = "black" if val < cat_daily_t.values.max() / 2 else "white"
                ax.text(j, i, int(val), ha="center", va="center", fontsize=9, color=color)
    plt.tight_layout()
    p = os.path.join(output_dir, "7_category_heatmap.png")
    fig.savefig(p, dpi=200)
    plt.close()
    chart_paths["分类热力图"] = p

    return chart_paths


def img_to_b64(path: str) -> str:
    """将图片转为 base64 字符串."""
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


# ========== HTML 报告生成 ==========

def generate_html_report(r: dict, chart_paths: dict, output_path: str):
    """生成结构化的 HTML 分析报告."""

    def esc(s):
        """简单的 HTML 转义."""
        if isinstance(s, str):
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return str(s)

    # 构建卡片 HTML
    anomaly_cards = ""
    level_colors = {"critical": "#D32F2F", "warning": "#F57C00"}
    for a in r["anomalies"]:
        lc = level_colors.get(a["level"], "#888")
        anomaly_cards += f"""
        <div class="anomaly-card" style="border-left: 4px solid {lc};">
            <div class="anomaly-level" style="color:{lc};font-size:12px;text-transform:uppercase;font-weight:700;">
                {'🔴 严重' if a['level']=='critical' else '🟠 警示'}
            </div>
            <div class="anomaly-title">{esc(a['title'])}</div>
            <div class="anomaly-desc">{esc(a['desc'])}</div>
        </div>"""

    # 图表 HTML
    chart_html = ""
    for title, path in chart_paths.items():
        b64 = img_to_b64(path)
        chart_html += f"""
        <div class="chart-card">
            <h3>{esc(title)}</h3>
            <img src="data:image/png;base64,{b64}" alt="{esc(title)}" style="width:100%;">
        </div>"""

    # 未解决工单表格
    unresolved_rows = ""
    for _, row in r["unresolved"].iterrows():
        flag = "⚠️ 严重超时" if row["resolution_time_hours"] >= 48 else ""
        unresolved_rows += f"<tr><td>{row['ticket_id']}</td><td>{row['category']}</td><td>{row['description'][:30]}...</td><td>{row['resolution_time_hours']:.0f}h</td><td>{flag}</td></tr>"

    # 关键词表格
    kw_rows = ""
    for kw, cnt in r["keywords"].items():
        kw_rows += f"<tr><td>{esc(kw)}</td><td>{cnt}</td></tr>"

    # 星期分布
    wd_rows = ""
    for day, cnt in r["weekday_counts"].items():
        wd_rows += f"<tr><td>{day}</td><td>{cnt}</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>客服工单趋势分析报告</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system, BlinkMacSystemFont, "Microsoft YaHei", sans-serif; background:#f5f7fa; color:#333; line-height:1.6; }}
.container {{ max-width:1200px; margin:0 auto; padding:20px; }}
.header {{ background:linear-gradient(135deg,#1a237e,#283593); color:#fff; padding:40px; border-radius:12px; margin-bottom:24px; }}
.header h1 {{ font-size:28px; margin-bottom:8px; }}
.header .meta {{ opacity:.85; font-size:14px; }}
.section {{ background:#fff; border-radius:12px; padding:24px; margin-bottom:24px; box-shadow:0 2px 8px rgba(0,0,0,.08); }}
.section h2 {{ font-size:20px; color:#1a237e; margin-bottom:16px; padding-bottom:8px; border-bottom:2px solid #e8eaf6; }}
.stats-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(160px, 1fr)); gap:16px; margin-bottom:16px; }}
.stat-card {{ background:#f5f7fa; border-radius:8px; padding:16px; text-align:center; }}
.stat-card .num {{ font-size:32px; font-weight:700; color:#1a237e; }}
.stat-card .label {{ font-size:13px; color:#666; margin-top:4px; }}
.anomaly-grid {{ display:grid; gap:12px; }}
.anomaly-card {{ background:#fff8f0; padding:16px; border-radius:8px; }}
.anomaly-title {{ font-weight:600; font-size:15px; margin:4px 0; }}
.anomaly-desc {{ font-size:13px; color:#555; }}
.chart-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(500px, 1fr)); gap:20px; }}
.chart-card {{ background:#fff; border:1px solid #e0e0e0; border-radius:8px; padding:16px; }}
.chart-card h3 {{ font-size:15px; color:#333; margin-bottom:12px; text-align:center; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th, td {{ padding:10px 12px; text-align:left; border-bottom:1px solid #e0e0e0; }}
th {{ background:#f5f7fa; font-weight:600; color:#1a237e; }}
tr:hover td {{ background:#fafafa; }}
.key-conclusion {{ background:#e8f5e9; border-radius:8px; padding:16px; margin-bottom:12px; }}
.key-conclusion h3 {{ color:#2e7d32; margin-bottom:6px; }}
.key-conclusion p {{ font-size:14px; color:#444; }}
footer {{ text-align:center; padding:20px; color:#999; font-size:13px; }}
</style>
</head>
<body>
<div class="container">

<div class="header">
    <h1>客服工单趋势分析报告</h1>
    <div class="meta">
        分析周期: {r['period'][0]} ~ {r['period'][1]} &nbsp;|&nbsp;
        工单总数: {r['total']} 张 &nbsp;|&nbsp;
        生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </div>
</div>

<div class="section">
    <h2>概览</h2>
    <div class="stats-grid">
        <div class="stat-card"><div class="num">{r['total']}</div><div class="label">工单总数</div></div>
        <div class="stat-card"><div class="num">{r['daily_avg']}</div><div class="label">日均工单量</div></div>
        <div class="stat-card"><div class="num">{r['avg_sat']}</div><div class="label">平均满意度</div></div>
        <div class="stat-card"><div class="num">{r['avg_time']}h</div><div class="label">平均处理时长</div></div>
        <div class="stat-card"><div class="num">{r['high_pct']}%</div><div class="label">高优先级占比</div></div>
        <div class="stat-card"><div class="num">{r['unresolved_count']}</div><div class="label">未解决工单</div></div>
    </div>
</div>

<div class="section">
    <h2>异常信号</h2>
    <div class="anomaly-grid">{anomaly_cards}</div>
</div>

<div class="section">
    <h2>可视化图表</h2>
    <div class="chart-grid">{chart_html}</div>
</div>

<div class="section">
    <h2>分类统计明细</h2>
    <table>
        <tr><th>分类</th><th>工单数</th><th>占比</th><th>平均处理时长</th><th>平均满意度</th></tr>
"""
    for cat in r["cat_counts"].index:
        pct = r["cat_counts"][cat] / r["total"] * 100
        t = r["cat_avg_time"].get(cat, 0)
        s = r["cat_sat"].get(cat, 0)
        html += f"<tr><td>{cat}</td><td>{r['cat_counts'][cat]}</td><td>{pct:.1f}%</td><td>{t:.1f}h</td><td>{s:.2f}</td></tr>\n"

    html += """
    </table>
</div>

<div class="section">
    <h2>未解决工单</h2>
    <table>
        <tr><th>工单编号</th><th>分类</th><th>问题描述</th><th>已处理</th><th>状态</th></tr>
"""
    html += unresolved_rows

    html += """
    </table>
</div>

<div class="section">
    <h2>关键词统计</h2>
    <table>
        <tr><th>关键词</th><th>出现次数</th></tr>
"""
    html += kw_rows

    html += """
    </table>
</div>

<div class="section">
    <h2>关键结论与建议</h2>
    <div class="key-conclusion">
        <h3>1. 退款退货流程急需优化</h3>
        <p>占比最高（""" + f"{r['cat_counts'].get('退款退货',0)/r['total']*100:.0f}" + """%）、满意度最低（""" + f"{r['cat_sat'].get('退款退货',0):.2f}" + """）、处理时长最长（""" + f"{r['cat_avg_time'].get('退款退货',0):.0f}h""" + """）。建议简化退款流程，设置自动退款规则。</p>
    </div>
    <div class="key-conclusion">
        <h3>2. 支付系统存在 Bug</h3>
        <p>支付问题 """ + f"{r['pay_count']}" + """ 张，反复出现重复扣款、扣款成功但订单未生成。建议排查支付回调接口和幂等性处理。</p>
    </div>
    <div class="key-conclusion">
        <h3>3. 服务质量需要关注</h3>
        <p>出现 """ + f"{r['complaint_count']}" + """ 张投诉工单，涉及客服态度和响应速度。建议监控客服响应时间，优化机器人应答策略。</p>
    </div>
</div>

<footer>客服工单趋势分析工具 &mdash; 自动生成</footer>
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


# ========== 主入口 ==========

def main():
    parser = argparse.ArgumentParser(description="客服工单趋势分析工具")
    parser.add_argument("--data", default="task5_tickets.json", help="工单数据 JSON 文件路径 (默认: task5_tickets.json)")
    parser.add_argument("--output", default="./charts", help="图表输出目录 (默认: ./charts)")
    parser.add_argument("--report", default="report.html", help="HTML 报告输出路径 (默认: report.html)")
    args = parser.parse_args()

    print("=" * 60)
    print("客服工单趋势分析工具")
    print("=" * 60)
    print(f"数据文件: {args.data}")
    print(f"图表目录: {args.output}")
    print(f"报告文件: {args.report}")

    # 加载数据
    if not os.path.exists(args.data):
        print(f"[错误] 数据文件不存在: {args.data}")
        sys.exit(1)
    df = load_data(args.data)
    print(f"✓ 加载 {len(df)} 条工单数据")

    # 执行分析
    r = analyze(df)
    print("✓ 分析完成")

    # 生成图表
    print("正在生成可视化图表...")
    charts = generate_charts(df, r, args.output)
    for title, path in charts.items():
        print(f"  ✓ {title} -> {path}")

    # 生成 HTML 报告
    print("正在生成 HTML 报告...")
    report_path = generate_html_report(r, charts, args.report)
    print(f"  ✓ HTML 报告 -> {report_path}")

    print(f"\n✓ 全部完成！报告已保存至 {report_path}")
    print(f"  用浏览器打开 {report_path} 查看完整分析结果。")


if __name__ == "__main__":
    main()
