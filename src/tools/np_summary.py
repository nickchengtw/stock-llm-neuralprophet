
import seaborn as sns
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('reports/summary.csv', index_col=0)
# df.drop(columns=['is_lower', 'with_future'], inplace=True)
df.rename(columns={
    "default": "預設參數",
    "add_period": "季節性",
    "add_ar": "自回歸",
    "add_holidays": "特殊節日",
    "lag_vol_price": "成交量與移動平均線",
    "lag_inst": "三大法人買賣超",
    "lag_share": "大股東持股比例",
    }, inplace=True)


plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 修改中文字體
plt.rcParams['axes.unicode_minus'] = False # 顯示負號

# Melt into long format
df_melted = (df / 100).melt(var_name="Category", value_name="Value")

# Set figure size (wider and taller for clarity)
plt.figure(figsize=(9, 6))

colors = [
    "#FF9999",  # light red
    "#99FF99",  # light green
    "#9999FF",  # light blue
    "#FFCC99",  # light orange
    "#CC99FF",  # light purple
    "#FFFF99",  # light yellow
    "#66CCCC"   # teal
]

# Horizontal box plots
ax = sns.boxplot(
    data=df_melted,
    y="Value",  # Categories on Y-axis
    x="Category",     # Values on X-axis
    order=df.columns,  # Keep original order
    palette=colors
)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))  # xmax=1 if data in 0-1 range

plt.ylabel("MAPE誤差")
plt.xlabel("調整超參數的各階段")
plt.title("NeuralProphet 模型完成各階段超參數調整後的預測誤差分布")
plt.xticks(rotation=45)
plt.tight_layout()

df.describe()
plt.show()
