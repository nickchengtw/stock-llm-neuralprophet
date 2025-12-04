import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# TODO this table is manually created, need to generate it automatically
df = pd.read_csv('reports/llm_result.csv', index_col=0)
df.drop(columns=['with RAG', 'news RAG r15', 'news RAG r20', 'human rule', 'use summary'], inplace=True)
df.rename(columns={
    "no RAG": "Llama-3-Taiwan",
    "news RAG cr15": "Llama-3-Taiwan + RAG",
    "deepseek no RAG": "DeepSeek V3.1",
    "Deepseek-chat-v3-0324": "DeepSeek V3.1 + RAG",
    }, inplace=True)

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 修改中文字體
plt.rcParams['axes.unicode_minus'] = False # 顯示負號

# Melt into long format
df_melted = (df / 100).melt(var_name="Category", value_name="Value")

# Set figure size (wider and taller for clarity)
plt.figure(figsize=(7, 5))

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
plt.xlabel("預測方式")
plt.title("LLM各種預測方式比較")
plt.tight_layout()

print(df.describe())
plt.show()
