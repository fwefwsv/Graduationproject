#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阿尔兹海默症预测模型训练脚本
基于XGBoost的机器学习模型训练和优化
来源：https://pythonhacker.blog.csdn.net/article/details/141941961
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data():
    """加载和预处理数据"""
    print("正在加载数据...")
    df = pd.read_csv("alzheimers_disease_data.csv")
    df = df.drop(columns=['PatientID', 'DoctorInCharge'], errors='ignore')
    # 列名汉化映射
    column_mapping = {
        "Age": "年龄", "Gender": "性别", "Ethnicity": "种族", "EducationLevel": "教育水平",
        "BMI": "身体质量指数（BMI）", "Smoking": "吸烟状况", "AlcoholConsumption": "酒精摄入量",
        "PhysicalActivity": "体育活动时间", "DietQuality": "饮食质量评分", "SleepQuality": "睡眠质量评分",
        "FamilyHistoryAlzheimers": "家族阿尔茨海默病史", "CardiovascularDisease": "心血管疾病",
        "Diabetes": "糖尿病", "Depression": "抑郁症史", "HeadInjury": "头部受伤", "Hypertension": "高血压",
        "SystolicBP": "收缩压", "DiastolicBP": "舒张压", "CholesterolTotal": "胆固醇总量",
        "CholesterolLDL": "低密度脂蛋白胆固醇（LDL）", "CholesterolHDL": "高密度脂蛋白胆固醇（HDL）",
        "CholesterolTriglycerides": "甘油三酯", "MMSE": "简易精神状态检查（MMSE）得分",
        "FunctionalAssessment": "功能评估得分", "MemoryComplaints": "记忆抱怨", "BehavioralProblems": "行为问题",
        "ADL": "日常生活活动（ADL）得分", "Confusion": "混乱与定向障碍", "Disorientation": "迷失方向",
        "PersonalityChanges": "人格变化", "DifficultyCompletingTasks": "完成任务困难", "Forgetfulness": "健忘",
        "Diagnosis": "诊断状态", "DoctorInCharge": "主诊医生"
    }
    df.rename(columns=column_mapping, inplace=True)
    
    # 处理分类变量
    categorical_columns = ['种族']
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    return df_encoded

def perform_eda(df):
    """执行探索性数据分析"""
    print("正在执行数据探索分析...")
    
    # 创建可视化目录
    import os
    os.makedirs('static/images', exist_ok=True)
    
    # 1. 诊断状态分布
    plt.figure(figsize=(10, 6))
    diagnosis_counts = df['诊断状态'].value_counts()
    plt.pie(diagnosis_counts.values, labels=['未确诊', '确诊'], autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'])
    plt.title('阿尔兹海默症诊断状态分布')
    plt.savefig('static/images/diagnosis_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 年龄分布
    plt.figure(figsize=(12, 6))
    df.boxplot(column='年龄', by='诊断状态', ax=plt.gca())
    plt.title('年龄与阿尔兹海默症诊断状态的关系')
    plt.suptitle('')
    plt.ylabel('年龄')
    plt.savefig('static/images/age_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 相关性热力图
    plt.figure(figsize=(20, 16))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    plt.title('特征相关性热力图')
    plt.tight_layout()
    plt.savefig('static/images/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 重要特征分布
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # MMSE得分分布
    sns.violinplot(data=df, x='诊断状态', y='简易精神状态检查（MMSE）得分', ax=axes[0,0])
    axes[0,0].set_title('MMSE得分与诊断状态')
    
    # 功能评估得分分布
    sns.violinplot(data=df, x='诊断状态', y='功能评估得分', ax=axes[0,1])
    axes[0,1].set_title('功能评估得分与诊断状态')
    
    # ADL得分分布
    sns.violinplot(data=df, x='诊断状态', y='日常生活活动（ADL）得分', ax=axes[1,0])
    axes[1,0].set_title('ADL得分与诊断状态')
    
    # BMI分布
    sns.violinplot(data=df, x='诊断状态', y='身体质量指数（BMI）', ax=axes[1,1])
    axes[1,1].set_title('BMI与诊断状态')
    
    plt.tight_layout()
    plt.savefig('static/images/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("数据探索分析完成，图表已保存到 static/images/ 目录")

def balance_dataset(X, y):
    """使用采样技术平衡数据集"""
    print("正在平衡数据集...")
    
    # 先过采样少数类
    over_sampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
    X_over, y_over = over_sampler.fit_resample(X, y)
    
    # 再欠采样多数类
    under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = under_sampler.fit_resample(X_over, y_over)
    
    print(f"原始数据集大小: {len(X)}")
    print(f"平衡后数据集大小: {len(X_resampled)}")
    
    return X_resampled, y_resampled

def train_xgboost_model(X_train, y_train, X_valid, y_valid):
    """训练XGBoost模型"""
    print("正在训练XGBoost模型...")
    
    # 设置参数
    xgb_params = {
        'eta': 0.1,
        'colsample_bytree': 0.4,
        'max_depth': 8,
        'lambda': 2.0,
        'eval_metric': 'auc',
        'objective': 'binary:logistic',
        'nthread': -1,
        'silent': 1,
        'booster': 'gbtree'
    }
    
    # 创建DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns.tolist())
    dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=X_valid.columns.tolist())
    
    # 设置监控列表
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    
    # 训练模型
    model = xgb.train(
        xgb_params, 
        dtrain, 
        num_boost_round=4000,
        evals=watchlist,
        verbose_eval=100,
        early_stopping_rounds=100
    )
    
    return model

def evaluate_model(model, X_train, y_train, X_valid, y_valid, X_test, y_test):
    """评估模型性能"""
    print("正在评估模型性能...")
    
    # 创建DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns.tolist())
    dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=X_valid.columns.tolist())
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=X_test.columns.tolist())
    
    # 预测
    pred_train = model.predict(dtrain)
    pred_valid = model.predict(dvalid)
    pred_test = model.predict(dtest)
    
    # 计算AUC
    train_auc = roc_auc_score(y_train, pred_train)
    valid_auc = roc_auc_score(y_valid, pred_valid)
    test_auc = roc_auc_score(y_test, pred_test)
    
    print(f"训练集 AUC: {train_auc:.4f}")
    print(f"验证集 AUC: {valid_auc:.4f}")
    print(f"测试集 AUC: {test_auc:.4f}")
    
    # 绘制ROC曲线
    from sklearn.metrics import roc_curve
    
    plt.figure(figsize=(10, 8))
    
    # 测试集ROC曲线
    fpr_test, tpr_test, _ = roc_curve(y_test, pred_test)
    plt.plot(fpr_test, tpr_test, label=f'测试集 (AUC = {test_auc:.3f})', linewidth=2)
    
    # 验证集ROC曲线
    fpr_valid, tpr_valid, _ = roc_curve(y_valid, pred_valid)
    plt.plot(fpr_valid, tpr_valid, label=f'验证集 (AUC = {valid_auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('假正率')
    plt.ylabel('真正率')
    plt.title('ROC曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('static/images/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 混淆矩阵
    y_pred = (pred_test > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig('static/images/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 特征重要性
    importance = model.get_score(importance_type='weight')
    importance_df = pd.DataFrame({
        'feature': list(importance.keys()),
        'importance': list(importance.values())
    }).sort_values('importance', ascending=False).head(15)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, y='feature', x='importance')
    plt.title('特征重要性 (前15个)')
    plt.xlabel('重要性得分')
    plt.tight_layout()
    plt.savefig('static/images/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'train_auc': train_auc,
        'valid_auc': valid_auc,
        'test_auc': test_auc,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

def main():
    """主函数"""
    print("=" * 60)
    print("阿尔兹海默症智能分析预测系统 - 模型训练")
    print("=" * 60)
    
    # 1. 加载和预处理数据
    df = load_and_preprocess_data()
    
    # 2. 执行探索性数据分析
    perform_eda(df)
    
    # 3. 准备特征和标签
    feature_columns = [col for col in df.columns if col != '诊断状态']
    X = df[feature_columns]
    y = df['诊断状态']
    
    # 4. 划分数据集
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)
    
    print(f"\n数据集划分:")
    print(f"训练集: {len(X_train)} 样本")
    print(f"验证集: {len(X_valid)} 样本")
    print(f"测试集: {len(X_test)} 样本")
    
    # 5. 平衡数据集
    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)
    
    # 6. 训练模型
    model = train_xgboost_model(X_train_balanced, y_train_balanced, X_valid, y_valid)
    
    # 7. 评估模型
    results = evaluate_model(model, X_train_balanced, y_train_balanced, X_valid, y_valid, X_test, y_test)
    
    # 8. 保存模型
    joblib.dump(model, 'alzheimer_model.pkl')
    joblib.dump(X.columns.tolist(), 'feature_columns.pkl')
    
    print("\n" + "=" * 60)
    print("模型训练完成！")
    print(f"模型已保存为: alzheimer_model.pkl")
    print(f"特征列已保存为: feature_columns.pkl")
    print("=" * 60)

if __name__ == "__main__":
    main()