import pandas as pd


df = pd.read_csv("./alzheimers_disease_data.csv")

## 列名汉化
df.rename(columns={"Age": "年龄", "Gender": "性别", "Ethnicity": "种族", "EducationLevel": "教育水平",
                   "BMI": "身体质量指数（BMI）", "Smoking": "吸烟状况", "AlcoholConsumption": "酒精摄入量",
                   "PhysicalActivity": "体育活动时间", "DietQuality": "饮食质量评分", "SleepQuality": "睡眠质量评分",
                   "FamilyHistoryAlzheimers": "家族阿尔茨海默病史", "CardiovascularDisease": "心血管疾病",
                   "Diabetes": "糖尿病", "Depression": "抑郁症史", "HeadInjury": "头部受伤", "Hypertension": "高血压",
                   "SystolicBP": "收缩压", "DiastolicBP": "舒张压", "CholesterolTotal": "胆固醇总量",
                   "CholesterolLDL": "低密度脂蛋白胆固醇（LDL）", "CholesterolHDL": "高密度脂蛋白胆固醇（HDL）",
                   "CholesterolTriglycerides": "甘油三酯", "MMSE": "简易精神状态检查（MMSE）得分",
                   "FunctionalAssessment": "功能评估得分", "MemoryComplaints": "记忆抱怨",
                   "BehavioralProblems": "行为问题", "ADL": "日常生活活动（ADL）得分", "Confusion": "混乱与定向障碍",
                   "Disorientation": "迷失方向", "PersonalityChanges": "人格变化",
                   "DifficultyCompletingTasks": "完成任务困难", "Forgetfulness": "健忘", "Diagnosis": "诊断状态",
                   "DoctorInCharge": "主诊医生"}, inplace=True)

# df.info()
tmp = df["诊断状态"].value_counts().to_frame().reset_index().rename(columns={"count": "数量"})

tmp["诊断状态"] = tmp["诊断状态"].map(lambda x: "是" if x == 1 else "否")
tmp["百分比"] = tmp["数量"].map(lambda x: round(x / tmp["数量"].sum() * 100, 2))

labels, values, percent = tmp["诊断状态"].tolist(), tmp["数量"].tolist(), tmp["百分比"].tolist()

import matplotlib.pyplot as plt

# ========== 你的数据（已经准备好了）==========
# labels, values, percent 已经在你的代码中定义好了
# 这里直接复用：

fig, ax = plt.subplots(figsize=(9, 9), facecolor='white')

# 颜色配置
colors = ['#4A90D9', '#E74C3C']      # 蓝色=未患病，红色=患病
explode = (0.02, 0.08)               # 突出显示"患病"部分

# 绘制环形图
wedges, texts, autotexts = ax.pie(
    values,
    labels=labels,
    autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100.*sum(values))}人)',
    startangle=90,
    colors=colors,
    explode=explode,
    shadow=True,
    textprops={'fontsize': 14, 'fontweight': 'bold'},
    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2)
)

# 美化百分比文字
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(13)
    autotext.set_fontweight('bold')

# 标题
ax.set_title('阿尔茨海默病诊断状态分布', fontsize=18, fontweight='bold', pad=20, color='#2C3E50')

# 图例
legend_labels = [f'未患病 ({values[0]}人, {percent[0]}%)',
                 f'患病 ({values[1]}人, {percent[1]}%)']
ax.legend(wedges, legend_labels, title="诊断状态", loc="center", fontsize=11,
          title_fontsize=12, frameon=True, fancybox=True, shadow=True)

# 中心显示总人数
ax.text(0, 0, f'总计\n{sum(values)}人', ha='center', va='center', fontsize=16,
        fontweight='bold', color='#2C3E50')
import seaborn as sns
plt.tight_layout()
# plt.show()
fig,ax = plt.subplots(1,1,figsize=(20, 16))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(df.corr(), annot= True, cmap=cmap, vmax=.5, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.set_xticklabels(labels=df.columns, rotation=90, fontsize=12)
ax.set_yticklabels(labels=df.columns, rotation=00, fontsize=12)
plt.show()
