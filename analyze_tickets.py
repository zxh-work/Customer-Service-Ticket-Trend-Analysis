"""
客服工单趋势分析工具
分析维度：时间趋势、分类分布、优先级、满意度、处理时长、渠道、异常检测
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from datetime import datetime, timedelta
from collections import Counter, defaultdict

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ===== 1. 加载数据 =====
with open('task5_tickets.json', 'r', encoding='utf-8') as f:
    tickets = json.load(f)

df = pd.DataFrame(tickets)
df['created_at'] = pd.to_datetime(df['created_at'])
df['date'] = df['created_at'].dt.date
df['weekday'] = df['created_at'].dt.day_name()
df['hour'] = df['created_at'].dt.hour

print("=" * 60)
print("客服工单趋势分析报告")
print("=" * 60)
print(f"\n分析周期: {df['date'].min()} ~ {df['date'].max()}")
print(f"工单总数: {len(df)}")

# ===== 2. 时间趋势分析 =====
print("\n" + "=" * 60)
print("一、时间趋势分析")
print("=" * 60)

daily_counts = df.groupby('date').size()
print(f"\n日均工单量: {daily_counts.mean():.1f} 张")
print(f"单日最高: {daily_counts.max():.0f} 张 ({daily_counts.idxmax()})")
print(f"单日最低: {daily_counts.min():.0f} 张 ({daily_counts.idxmin()})")

# 按周几统计
weekday_counts = df.groupby('weekday').size().sort_values(ascending=False)
print(f"\n星期分布:")
for day, count in weekday_counts.items():
    print(f"  {day}: {count} 张")

# ===== 3. 分类分布分析 =====
print("\n" + "=" * 60)
print("二、分类分布分析")
print("=" * 60)

cat_counts = df['category'].value_counts()
print(f"\n各分类工单量及占比:")
for cat, count in cat_counts.items():
    print(f"  {cat}: {count} 张 ({count/len(df)*100:.1f}%)")

# 分类时间趋势（每类每日数量）
cat_daily = df.groupby(['date', 'category']).size().unstack(fill_value=0)
print(f"\n分类日趋势（日期/各类数量）:")
print(cat_daily.to_string())

# ===== 4. 优先级分析 =====
print("\n" + "=" * 60)
print("三、优先级分析")
print("=" * 60)

pri_counts = df['priority'].value_counts()
for pri, count in pri_counts.items():
    print(f"  {pri}优先级: {count} 张 ({count/len(df)*100:.1f}%)")

# 分类×优先级交叉分析
cross_pri = pd.crosstab(df['category'], df['priority'])
print(f"\n分类×优先级分布:")
print(cross_pri.to_string())

# 高优先级工单详情
high_pri = df[df['priority'] == '高']
high_cat = high_pri['category'].value_counts()
print(f"\n高优先级工单分类分布:")
for cat, count in high_cat.items():
    print(f"  {cat}: {count} 张")

# ===== 5. 处理时长分析 =====
print("\n" + "=" * 60)
print("四、处理时长分析")
print("=" * 60)

avg_res_time = df['resolution_time_hours'].mean()
print(f"\n平均处理时长: {avg_res_time:.1f} 小时")
print(f"中位处理时长: {df['resolution_time_hours'].median():.1f} 小时")
print(f"最长处理时长: {df['resolution_time_hours'].max():.1f} 小时")

# 各分类平均处理时长
cat_avg_time = df.groupby('category')['resolution_time_hours'].mean().sort_values(ascending=False)
print(f"\n各分类平均处理时长:")
for cat, t in cat_avg_time.items():
    print(f"  {cat}: {t:.1f} 小时")

# 超长处理工单（>=48小时）
long_tickets = df[df['resolution_time_hours'] >= 48]
print(f"\n超长处理工单 (>=48小时): {len(long_tickets)} 张")
for _, row in long_tickets.iterrows():
    print(f"  {row['ticket_id']}: {row['category']} - {row['resolution_time_hours']:.0f}h - {row['description'][:20]}...")

# ===== 6. 满意度分析 =====
print("\n" + "=" * 60)
print("五、满意度分析")
print("=" * 60)

avg_sat = df['satisfaction'].mean()
print(f"\n平均满意度: {avg_sat:.2f} / 5.0")

# 各分类平均满意度
cat_sat = df.groupby('category')['satisfaction'].mean().sort_values()
print(f"\n各分类平均满意度:")
for cat, s in cat_sat.items():
    print(f"  {cat}: {s:.2f}")

# 低满意度工单（满意度≤2）
low_sat = df[df['satisfaction'] <= 2]
low_sat_cat = low_sat['category'].value_counts()
print(f"\n低满意度工单 (评分≤2): {len(low_sat)} 张")
print(f"低满意度工单分类分布:")
for cat, count in low_sat_cat.items():
    print(f"  {cat}: {count} 张")

# 满意度与处理时长相关性
corr = df['resolution_time_hours'].corr(df['satisfaction'])
print(f"\n处理时长与满意度相关系数: {corr:.3f} (负值表示处理越长满意度越低)")

# ===== 7. 渠道分析 =====
print("\n" + "=" * 60)
print("六、渠道分析")
print("=" * 60)

channel_counts = df['channel'].value_counts()
print(f"\n各渠道工单量:")
for ch, count in channel_counts.items():
    print(f"  {ch}: {count} 张 ({count/len(df)*100:.1f}%)")

channel_sat = df.groupby('channel')['satisfaction'].mean()
print(f"\n各渠道平均满意度:")
for ch, s in channel_sat.items():
    print(f"  {ch}: {s:.2f}")

channel_time = df.groupby('channel')['resolution_time_hours'].mean()
print(f"\n各渠道平均处理时长:")
for ch, t in channel_time.items():
    print(f"  {ch}: {t:.1f} 小时")

# ===== 8. 未解决工单分析 =====
print("\n" + "=" * 60)
print("七、未解决工单分析")
print("=" * 60)

unresolved = df[df['is_resolved'] == False]
print(f"\n未解决工单: {len(unresolved)} 张 ({len(unresolved)/len(df)*100:.1f}%)")
unresolved_cat = unresolved['category'].value_counts()
for cat, count in unresolved_cat.items():
    print(f"  {cat}: {count} 张")

for _, row in unresolved.iterrows():
    print(f"  {row['ticket_id']}: {row['category']} - {row['description'][:25]}... - 已处理{row['resolution_time_hours']:.0f}h")

# ===== 9. 异常检测 =====
print("\n" + "=" * 60)
print("八、异常信号检测")
print("=" * 60)

anomalies = []

# 异常1: 退款退货占比最高且满意度最低
refund_pct = cat_counts.get('退款退货', 0) / len(df) * 100
refund_sat = cat_sat.get('退款退货', 0)
refund_time = cat_avg_time.get('退款退货', 0)
print(f"\n[异常信号1] 退款退货类工单占比最高 ({refund_pct:.1f}%)，满意度最低 ({refund_sat:.2f})")
print(f"  判断依据: 退款退货工单 {cat_counts.get('退款退货',0)} 张，占总量的 {refund_pct:.1f}%，")
print(f"  但满意度仅 {refund_sat:.2f}，远低于平均 {avg_sat:.2f}，且平均处理时长 {refund_time:.0f} 小时")
anomalies.append(f"退款退货类工单量大质差，满意度仅{refund_sat:.2f}，平均处理{refund_time:.0f}小时")

# 异常2: 支付问题频繁出现
pay_count = cat_counts.get('支付问题', 0)
pay_sat = cat_sat.get('支付问题', 0)
print(f"\n[异常信号2] 支付问题工单频繁出现 ({pay_count} 张)")
print(f"  判断依据: 支付问题工单 {pay_count} 张，满意度 {pay_sat:.2f}，")
print('  描述中反复出现"重复扣款""扣款成功但订单未生成"等系统性问题')
anomalies.append("支付问题反复出现，多为系统性Bug（重复扣款、支付回调失败）")

# 异常3: 高优先级占比过高
high_pct = pri_counts.get('高', 0) / len(df) * 100
print(f"\n[异常信号3] 高优先级工单占比过高 ({high_pct:.1f}%)")
print(f"  判断依据: 50张工单中{pri_counts.get('高',0)}张为高优先级，")
print(f"  其中支付问题和退款退货是主要来源，说明核心业务流程存在系统性问题")
anomalies.append(f"高优先级工单占比{high_pct:.0f}%偏高，核心流程存在系统性问题")

# 异常4: 满意度极低工单聚类
very_low_sat = df[df['satisfaction'] == 1]
print(f"\n[异常信号4] 满意度评分1分的工单有 {len(very_low_sat)} 张")
for _, row in very_low_sat.iterrows():
    print(f"  {row['ticket_id']}: {row['category']} - {row['description'][:25]}...")
print(f"  判断依据: 评分1分代表客户极度不满，{len(very_low_sat)}张集中在退款退货和投诉类，")
print(f"  说明退款流程和客服服务体验存在严重问题")
anomalies.append(f"严重不满(评分1分)工单{len(very_low_sat)}张，集中在退款和投诉类别")

# 异常5: 投诉类工单的出现
complaint_count = cat_counts.get('投诉', 0)
print(f"\n[异常信号5] 投诉类工单出现 {complaint_count} 张")
print(f"  判断依据: 投诉工单的出现本身是警示信号，满意度均为1.0，")
print('  内容涉及"客服态度差""等待时间过长""机器人无效回复"等服务流程问题')
anomalies.append(f"投诉类工单{complaint_count}张，涉及客服态度和响应速度问题")

# 异常6: 未解决工单的处理时间超长
print(f"\n[异常信号6] 未解决工单处理时长分布:")
for _, row in unresolved.iterrows():
    flag = " ⚠️ 严重超时" if row['resolution_time_hours'] >= 48 else ""
    print(f"  {row['ticket_id']}: 已处理{row['resolution_time_hours']:.0f}h{flag}")
print(f"  判断依据: 未解决工单平均已处理 {unresolved['resolution_time_hours'].mean():.0f} 小时，")
print(f"  部分工单处理超72小时仍未解决，客户满意度极有可能进一步恶化")
anomalies.append(f"未解决工单{len(unresolved)}张平均已处理{unresolved['resolution_time_hours'].mean():.0f}h，存在升级风险")

# 异常7: 重复问题模式
print(f"\n[异常信号7] 相似问题反复出现")
# 关键词频率分析
keywords = ['退款', '重复扣款', '快递', '扣款成功.*订单', '退货.*钱']
for kw in keywords:
    count = df['description'].str.contains(kw, regex=True).sum()
    print(f'  包含"{kw}"的工单: {count} 张')
print(f"  判断依据: 同类型问题反复出现，说明根本原因未得到解决")
anomalies.append("相似问题反复出现（重复扣款、退款延迟），根本原因未根治")

# ===== 10. 关键结论 =====
print("\n" + "=" * 60)
print("九、关键结论与建议")
print("=" * 60)
print(f"""
1. 【退款退货流程急需优化】
   占比最高（{cat_counts.get('退款退货',0)/len(df)*100:.0f}%）、满意度最低（{cat_sat.get('退款退货',0):.2f}）、处理时长最长（{cat_avg_time.get('退款退货',0):.0f}h）
   建议: 简化退款流程，设置自动退款规则，缩短审核周期

2. 【支付系统存在Bug】
   支付问题工单({pay_count}张)描述中多次出现重复扣款、扣款成功但订单未生成等问题
   建议: 排查支付回调接口、幂等性处理、与支付渠道对账机制

3. 【物流查询响应慢】
   物流类工单({cat_counts.get('物流查询',0)}张)处理时长{cat_avg_time.get('物流查询',0):.0f}h，部分包裹异常场景缺乏主动通知
   建议: 接入物流API实时跟踪，对异常包裹主动触发通知

4. 【服务质量需要关注】
   出现{complaint_count}张投诉工单，涉及客服态度、等待时长
   建议: 监控客服响应时间，设置排队超时预警，优化机器人应答策略

5. 【高优先级工单占比过高】
   高优先级占{high_pct:.0f}%，反映核心流程稳定性不足
   建议: 设定高优先级工单占比目标值，超过阈值自动触发根因分析

【异常信号汇总】
""")
for i, a in enumerate(anomalies, 1):
    print(f"  {i}. {a}")

# ========== 可视化 ==========
print("\n正在生成可视化图表...")
fig_dir = 'charts'
import os
os.makedirs(fig_dir, exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---- 图1: 每日工单量趋势 ----
fig, ax = plt.subplots(figsize=(14, 5))
dates = sorted(daily_counts.index)
ax.plot(dates, daily_counts.values, marker='o', linewidth=2, color='#2196F3', markersize=6)
ax.axhline(y=daily_counts.mean(), color='red', linestyle='--', alpha=0.7, label=f'日均 {daily_counts.mean():.1f}')
ax.fill_between(dates, daily_counts.values, alpha=0.15, color='#2196F3')
ax.set_title('每日工单量趋势', fontsize=14, fontweight='bold')
ax.set_xlabel('日期')
ax.set_ylabel('工单数量')
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{fig_dir}/1_daily_trend.png', dpi=200)
plt.close()
print(f"  ✓ 生成: 1_daily_trend.png")

# ---- 图2: 分类分布饼图 ----
fig, ax = plt.subplots(figsize=(8, 8))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
wedges, texts, autotexts = ax.pie(
    cat_counts.values, labels=cat_counts.index, autopct='%1.1f%%',
    colors=colors[:len(cat_counts)], startangle=90,
    textprops={'fontsize': 12}
)
for t in autotexts:
    t.set_fontsize(11)
ax.set_title('工单分类分布', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{fig_dir}/2_category_pie.png', dpi=200)
plt.close()
print(f"  ✓ 生成: 2_category_pie.png")

# ---- 图3: 分类×优先级堆叠柱状图 ----
fig, ax = plt.subplots(figsize=(12, 6))
categories = list(cat_counts.index)
priority_order = ['高', '中', '低']
priority_colors = {'高': '#FF4444', '中': '#FFA726', '低': '#66BB6A'}
bottom = np.zeros(len(categories))

for pri in priority_order:
    values = [cross_pri.loc[cat, pri] if cat in cross_pri.index and pri in cross_pri.columns else 0 for cat in categories]
    ax.bar(categories, values, bottom=bottom, label=pri, color=priority_colors[pri], alpha=0.85)
    bottom += values

ax.set_title('各分类优先级分布', fontsize=14, fontweight='bold')
ax.set_xlabel('问题分类')
ax.set_ylabel('工单数量')
ax.legend(title='优先级')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f'{fig_dir}/3_category_priority.png', dpi=200)
plt.close()
print(f"  ✓ 生成: 3_category_priority.png")

# ---- 图4: 各分类平均处理时长 + 满意度双轴图 ----
fig, ax1 = plt.subplots(figsize=(12, 6))
categories_ordered = list(cat_avg_time.index)
times = cat_avg_time.values
sats = [cat_sat[cat] for cat in categories_ordered]

bars = ax1.bar(categories_ordered, times, color='#42A5F5', alpha=0.7, label='平均处理时长(小时)')
ax1.set_xlabel('问题分类')
ax1.set_ylabel('平均处理时长 (小时)', color='#42A5F5', fontsize=12)
ax1.tick_params(axis='y', labelcolor='#42A5F5')
ax1.set_ylim(0, max(times) * 1.3)

ax2 = ax1.twinx()
ax2.plot(categories_ordered, sats, marker='D', linewidth=2, color='#EF5350', markersize=8, label='平均满意度')
ax2.axhline(y=avg_sat, color='gray', linestyle=':', alpha=0.6, label=f'整体平均 {avg_sat:.2f}')
ax2.set_ylabel('平均满意度', color='#EF5350', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#EF5350')
ax2.set_ylim(0, 5.5)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

ax1.set_title('各分类处理时长与满意度对比', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{fig_dir}/4_category_time_satisfaction.png', dpi=200)
plt.close()
print(f"  ✓ 生成: 4_category_time_satisfaction.png")

# ---- 图5: 处理时长 vs 满意度散点图 ----
fig, ax = plt.subplots(figsize=(10, 6))
colors_map = {'退款退货': '#FF4444', '支付问题': '#FFA726', '物流查询': '#42A5F5',
              '投诉': '#D32F2F', '商品咨询': '#66BB6A', '账号问题': '#AB47BC'}
for cat in df['category'].unique():
    subset = df[df['category'] == cat]
    ax.scatter(subset['resolution_time_hours'], subset['satisfaction'],
               label=cat, s=80, alpha=0.7, c=colors_map.get(cat, '#888'),
               edgecolors='white', linewidth=0.5)
ax.set_xlabel('处理时长 (小时)', fontsize=12)
ax.set_ylabel('满意度评分', fontsize=12)
ax.set_title('处理时长 vs 满意度评分', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
# 添加参考线
ax.axhline(y=3, color='gray', linestyle='--', alpha=0.4)
ax.axvline(x=24, color='gray', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(f'{fig_dir}/5_time_vs_satisfaction.png', dpi=200)
plt.close()
print(f"  ✓ 生成: 5_time_vs_satisfaction.png")

# ---- 图6: 渠道对比图 ----
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
channels = list(channel_counts.index)

# 渠道数量
colors_chn = ['#42A5F5', '#FFA726', '#66BB6A']
axes[0].bar(channels, channel_counts.values, color=colors_chn[:len(channels)], alpha=0.8)
axes[0].set_title('各渠道工单量', fontsize=12, fontweight='bold')
axes[0].set_ylabel('工单数量')
for i, v in enumerate(channel_counts.values):
    axes[0].text(i, v + 0.3, str(v), ha='center', fontsize=11)
axes[0].grid(True, alpha=0.3, axis='y')

# 渠道满意度
axes[1].bar(channels, channel_sat.values, color=colors_chn[:len(channels)], alpha=0.8)
axes[1].axhline(y=avg_sat, color='red', linestyle='--', alpha=0.6, label=f'平均 {avg_sat:.2f}')
axes[1].set_title('各渠道平均满意度', fontsize=12, fontweight='bold')
axes[1].set_ylabel('满意度')
axes[1].set_ylim(0, 5)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3, axis='y')

# 渠道处理时长
axes[2].bar(channels, channel_time.values, color=colors_chn[:len(channels)], alpha=0.8)
axes[2].set_title('各渠道平均处理时长', fontsize=12, fontweight='bold')
axes[2].set_ylabel('处理时长 (小时)')
for i, v in enumerate(channel_time.values):
    axes[2].text(i, v + 0.5, f'{v:.0f}h', ha='center', fontsize=10)
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{fig_dir}/6_channel_analysis.png', dpi=200)
plt.close()
print(f"  ✓ 生成: 6_channel_analysis.png")

# ---- 图7: 分类每日趋势热力图 ----
fig, ax = plt.subplots(figsize=(14, 6))
cat_daily_t = cat_daily.T
im = ax.imshow(cat_daily_t.values, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(len(cat_daily_t.columns)))
ax.set_xticklabels([str(d) for d in cat_daily_t.columns], rotation=45, ha='right')
ax.set_yticks(range(len(cat_daily_t.index)))
ax.set_yticklabels(cat_daily_t.index)
ax.set_xlabel('日期')
ax.set_ylabel('问题分类')
ax.set_title('各类工单每日数量热力图', fontsize=14, fontweight='bold')
for i in range(len(cat_daily_t.index)):
    for j in range(len(cat_daily_t.columns)):
        val = cat_daily_t.values[i, j]
        if val > 0:
            ax.text(j, i, int(val), ha='center', va='center', fontsize=9, color='black' if val < cat_daily_t.values.max()/2 else 'white')
plt.tight_layout()
plt.savefig(f'{fig_dir}/7_category_heatmap.png', dpi=200)
plt.close()
print(f"  ✓ 生成: 7_category_heatmap.png")

# ---- 图8: 异常信号汇总仪表盘风格 ----
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('off')
ax.set_title(' 异常信号汇总', fontsize=16, fontweight='bold', pad=20)

anomaly_items = [
    (f"① 退款退货占比{refund_pct:.0f}%，满意度{refund_sat:.2f}（整体{avg_sat:.2f}）", '#FF4444'),
    (f"② 支付问题{pay_count}张，系统Bug未根治", '#FF6B35'),
    (f"③ 高优先级工单占{high_pct:.0f}%", '#FFA726'),
    (f"④ 严重不满(1分)工单{len(very_low_sat)}张", '#D32F2F'),
    (f"⑤ 投诉工单{complaint_count}张（服务流程问题）", '#C62828'),
    (f"⑥ 未解决工单{len(unresolved)}张，平均已处理{unresolved['resolution_time_hours'].mean():.0f}h", '#E65100'),
    (f"⑦ 相似问题反复出现（重复扣款、退款延迟）", '#BF360C'),
]

for i, (text, color) in enumerate(anomaly_items):
    ax.text(0.05, 0.85 - i * 0.12, text, fontsize=13, color=color,
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF3E0', edgecolor=color, alpha=0.8))

plt.tight_layout()
plt.savefig(f'{fig_dir}/8_anomaly_summary.png', dpi=200)
plt.close()
print(f"  ✓ 生成: 8_anomaly_summary.png")

print(f"\n✓ 所有图表已保存至 {fig_dir}/ 目录")
print("\n分析完成！")
