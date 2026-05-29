#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阿尔兹海默症预测模型训练脚本
基于XGBoost的机器学习模型训练和优化
来源：https://pythonhacker.blog.csdn.net/article/details/141941961
"""

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import joblib
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def baseline_comparison(xgboost_model, X_train_balanced, y_train_balanced, X_test, y_test):
    """
    基线模型对比实验
    对比AdaBoost、Gradient Boosting、Logistic Regression、SVM和XGBoost的性能
    使用AUC、准确率和F1-score三个评价指标
    """
    from sklearn.metrics import accuracy_score, f1_score
    
    print("\n" + "=" * 60)
    print("基线模型对比实验")
    print("=" * 60)

    # 创建基线模型列表
    models = {
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }

    # 存储各模型的评估结果（AUC、准确率、F1-score）
    results = {
        'AUC': {},
        'Accuracy': {},
        'F1-score': {}
    }

    # 训练并评估XGBoost模型（已传入）
    print("正在评估XGBoost模型...")
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=X_test.columns.tolist())
    xgb_pred_proba = xgboost_model.predict(dtest)
    xgb_pred = (xgb_pred_proba > 0.5).astype(int)
    
    xgb_auc = roc_auc_score(y_test, xgb_pred_proba)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_f1 = f1_score(y_test, xgb_pred)
    
    results['AUC']['XGBoost'] = xgb_auc
    results['Accuracy']['XGBoost'] = xgb_acc
    results['F1-score']['XGBoost'] = xgb_f1
    
    print(f"XGBoost - AUC: {xgb_auc:.4f}, Accuracy: {xgb_acc:.4f}, F1-score: {xgb_f1:.4f}")

    # 训练并评估基线模型
    for name, model in models.items():
        print(f"\n正在训练{name}...")
        model.fit(X_train_balanced, y_train_balanced)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        auc = roc_auc_score(y_test, y_pred_proba)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results['AUC'][name] = auc
        results['Accuracy'][name] = acc
        results['F1-score'][name] = f1
        
        print(f"{name} - AUC: {auc:.4f}, Accuracy: {acc:.4f}, F1-score: {f1:.4f}")

    # 打印对比表格
    print("\n" + "=" * 60)
    print("模型性能对比表")
    print("=" * 60)
    print(f"{'模型名称':<20} {'AUC':<10} {'准确率':<10} {'F1-score':<10}")
    print("-" * 50)
    all_models = list(results['AUC'].keys())
    for name in sorted(all_models, key=lambda x: results['AUC'][x], reverse=True):
        print(f"{name:<20} {results['AUC'][name]:<10.4f} {results['Accuracy'][name]:<10.4f} {results['F1-score'][name]:<10.4f}")
    print("=" * 60)

    # 绘制多指标对比柱状图
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    metrics = ['AUC', 'Accuracy', 'F1-score']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, metric in enumerate(metrics):
        sorted_items = sorted(results[metric].items(), key=lambda x: x[1], reverse=True)
        model_names = [item[0] for item in sorted_items]
        scores = [item[1] for item in sorted_items]
        bar_colors = ['#1f77b4' if name == 'XGBoost' else '#7f7f7f' for name in model_names]
        
        sns.barplot(x=scores, y=model_names, palette=bar_colors, ax=axes[i])
        axes[i].set_xlabel(metric, fontsize=12)
        axes[i].set_title(f'{metric}对比', fontsize=14, fontweight='bold')
        axes[i].set_xlim(0.5, 1.0)
        axes[i].grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for j, v in enumerate(scores):
            axes[i].text(v + 0.01, j, f'{v:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('static/images/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n模型对比图已保存为: static/images/model_comparison.png")

    return results

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

    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=False)

    return df_encoded


def feature_selection(df, target_column='诊断状态', corr_threshold=0.05, multicollinearity_threshold=0.9):
    """
    特征筛选函数
    1. 剔除与目标变量相关性过低的特征
    2. 剔除存在多重共线性的特征
    
    参数：
        df: 原始DataFrame
        target_column: 目标变量列名，默认为'诊断状态'
        corr_threshold: 与目标变量相关系数阈值，低于此值的特征被剔除
        multicollinearity_threshold: 特征间相关系数阈值，高于此值认为存在多重共线性
    
    返回：
        筛选后的DataFrame
    """
    print("\n" + "=" * 60)
    print("特征筛选")
    print("=" * 60)
    
    # 分离特征和目标变量
    features = df.drop(columns=[target_column])
    target = df[target_column]
    
    # 1. 计算各特征与目标变量的Pearson相关系数
    feature_correlations = features.corrwith(target).abs()
    print(f"\n原始特征数量: {len(features.columns)}")
    
    # 剔除与目标变量相关性过低的特征
    low_corr_features = feature_correlations[feature_correlations < corr_threshold].index.tolist()
    remaining_features = feature_correlations[feature_correlations >= corr_threshold].index.tolist()
    
    print(f"\n与目标变量相关性 < {corr_threshold} 的特征（将被剔除）:")
    if low_corr_features:
        for feat in low_corr_features:
            print(f"  - {feat}: 相关系数 = {feature_correlations[feat]:.4f}")
    else:
        print("  无")
    
    # 更新剩余特征
    df_filtered = df[remaining_features + [target_column]]
    
    # 2. 处理多重共线性
    removed_features = []
    if len(remaining_features) > 1:
        # 计算特征间相关系数矩阵
        corr_matrix = df_filtered[remaining_features].corr().abs()
        
        # 找出相关系数 > multicollinearity_threshold 的特征对
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > multicollinearity_threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        # 需要剔除的特征
        features_to_remove = set()
        
        for feat1, feat2, corr_val in high_corr_pairs:
            if feat1 not in features_to_remove and feat2 not in features_to_remove:
                # 比较两个特征与目标变量的相关性，剔除相关性较小的那个
                corr1 = feature_correlations[feat1]
                corr2 = feature_correlations[feat2]
                
                if corr1 < corr2:
                    features_to_remove.add(feat1)
                    removed_features.append(f"{feat1}（与{feat2}相关系数={corr_val:.4f}，与目标相关系数较小）")
                else:
                    features_to_remove.add(feat2)
                    removed_features.append(f"{feat2}（与{feat1}相关系数={corr_val:.4f}，与目标相关系数较小）")
        
        # 剔除存在多重共线性的特征
        if features_to_remove:
            remaining_features = [f for f in remaining_features if f not in features_to_remove]
            df_filtered = df[remaining_features + [target_column]]
            
            print(f"\n存在多重共线性的特征（相关系数 > {multicollinearity_threshold}，将被剔除）:")
            for feat_info in removed_features:
                print(f"  - {feat_info}")
    
    print(f"\n特征筛选完成！")
    print(f"保留的特征数量: {len(remaining_features)}")
    print(f"保留的特征列表: {remaining_features}")
    
    return df_filtered


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
    ax = df.boxplot(column='年龄', by='诊断状态')
    plt.title('年龄与阿尔兹海默症诊断状态的关系')
    plt.suptitle('')
    plt.ylabel('年龄')
    plt.xlabel('诊断状态')
    ax.set_xticklabels(['正常', '阿尔兹海默症'])
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
    sns.violinplot(data=df, x='诊断状态', y='简易精神状态检查（MMSE）得分', ax=axes[0, 0])
    axes[0, 0].set_title('MMSE得分与诊断状态')

    # 功能评估得分分布
    sns.violinplot(data=df, x='诊断状态', y='功能评估得分', ax=axes[0, 1])
    axes[0, 1].set_title('功能评估得分与诊断状态')

    # ADL得分分布
    sns.violinplot(data=df, x='诊断状态', y='日常生活活动（ADL）得分', ax=axes[1, 0])
    axes[1, 0].set_title('ADL得分与诊断状态')

    # BMI分布
    sns.violinplot(data=df, x='诊断状态', y='身体质量指数（BMI）', ax=axes[1, 1])
    axes[1, 1].set_title('BMI与诊断状态')

    plt.tight_layout()
    plt.savefig('static/images/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. 年龄分布堆叠图（按诊断状态分组）
    plt.figure(figsize=(16, 8))
    plt.rcParams.update({'font.size': 14})  # 增大字体
    # 按诊断状态分组绘制年龄直方图
    df_healthy = df[df['诊断状态'] == 0]
    df_diagnosed = df[df['诊断状态'] == 1]
    plt.hist([df_healthy['年龄'], df_diagnosed['年龄']], bins=20, stacked=True,
             label=['未确诊', '确诊'], color=['#66b3ff', '#ff9999'], edgecolor='white', linewidth=1.2)
    plt.title('年龄分布堆叠图（按诊断状态分组）', fontsize=18, fontweight='bold')
    plt.xlabel('年龄', fontsize=16, fontweight='bold')
    plt.ylabel('人数', fontsize=16, fontweight='bold')
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3, linewidth=1)
    plt.tick_params(axis='both', labelsize=12)
    plt.savefig('static/images/age_stacked_histogram.png', dpi=400, bbox_inches='tight')
    plt.close()
    plt.rcParams.update({'font.size': 10})  # 恢复默认字体

    # 6. 性别与诊断状态分组柱状图
    plt.figure(figsize=(14, 8))
    plt.rcParams.update({'font.size': 14})
    sns.countplot(data=df, x='性别', hue='诊断状态', palette=['#66b3ff', '#ff9999'],
                  edgecolor='black', linewidth=1.2, saturation=0.9)
    plt.title('性别与诊断状态分布', fontsize=18, fontweight='bold')
    plt.xlabel('性别', fontsize=16, fontweight='bold')
    plt.ylabel('人数', fontsize=16, fontweight='bold')
    plt.xticks([0, 1], ['男性', '女性'], fontsize=14)
    plt.legend(['未确诊', '确诊'], fontsize=14)
    plt.grid(True, alpha=0.3, axis='y', linewidth=1)
    plt.tick_params(axis='both', labelsize=12)
    plt.savefig('static/images/gender_diagnosis_bar.png', dpi=400, bbox_inches='tight')
    plt.close()
    plt.rcParams.update({'font.size': 10})

    # 7. 种族与诊断状态分组柱状图
    plt.figure(figsize=(16, 8))
    plt.rcParams.update({'font.size': 14})
    # 从独热编码列中重建原始种族列
    ethnicity_cols = [col for col in df.columns if col.startswith('种族_')]
    if ethnicity_cols:
        df_temp = df.copy()
        # 使用idxmax获取原始种族值
        df_temp['种族'] = df[ethnicity_cols].idxmax(axis=1).str.replace('种族_', '').astype(int)
        sns.countplot(data=df_temp, x='种族', hue='诊断状态', palette=['#66b3ff', '#ff9999'],
                      edgecolor='black', linewidth=1.2, saturation=0.9)
        plt.title('种族与诊断状态分布', fontsize=18, fontweight='bold')
        plt.xlabel('种族', fontsize=16, fontweight='bold')
        plt.ylabel('人数', fontsize=16, fontweight='bold')
        plt.xticks([0, 1, 2, 3], ['Caucasian', 'African American', 'Asian', 'Other'], fontsize=12)
        plt.legend(['未确诊', '确诊'], fontsize=14)
        plt.grid(True, alpha=0.3, axis='y', linewidth=1)
        plt.tick_params(axis='both', labelsize=12)
        plt.savefig('static/images/ethnicity_diagnosis_bar.png', dpi=400, bbox_inches='tight')
    plt.close()
    plt.rcParams.update({'font.size': 10})
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

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns.tolist())
    dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=X_valid.columns.tolist())
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

    # 添加evals_result字典记录训练过程
    evals_result = {}

    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=4000,
        evals=watchlist,
        evals_result=evals_result,  # 新增参数
        verbose_eval=100,
        early_stopping_rounds=100
    )

    # 绘制学习曲线
    from xgboost_plot import plot_learning_curve
    plot_learning_curve(evals_result)

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
    plt.savefig('static/images/roc_curve.png', dpi=600, bbox_inches='tight')
    plt.close()

    # 混淆矩阵
    y_pred = (pred_test > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig('static/images/confusion_matrix.png', dpi=600, bbox_inches='tight')
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
    plt.savefig('static/images/feature_importance.png', dpi=600, bbox_inches='tight')
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

    # 3. 特征筛选
    df = feature_selection(df)

    # 4. 准备特征和标签
    feature_columns = [col for col in df.columns if col != '诊断状态']
    X = df[feature_columns]
    y = df['诊断状态']

    # 5. 划分数据集
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42,
                                                          stratify=y_temp)

    print(f"\n数据集划分:")
    print(f"训练集: {len(X_train)} 样本")
    print(f"验证集: {len(X_valid)} 样本")
    print(f"测试集: {len(X_test)} 样本")

    # 6. 平衡数据集
    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)

    # 7. 训练模型
    model = train_xgboost_model(X_train_balanced, y_train_balanced, X_valid, y_valid)

    # 8. 评估模型
    results = evaluate_model(model, X_train_balanced, y_train_balanced, X_valid, y_valid, X_test, y_test)

    # 9. 保存模型
    joblib.dump(model, 'alzheimer_model.pkl')
    joblib.dump(X.columns.tolist(), 'feature_columns.pkl')
    # 10. 基线模型对比实验
    baseline_comparison(model, X_train_balanced, y_train_balanced, X_test, y_test)
    print("\n" + "=" * 60)
    print("模型训练完成！")
    print(f"模型已保存为: alzheimer_model.pkl")
    print(f"特征列已保存为: feature_columns.pkl")
    print("=" * 60)


if __name__ == "__main__":
    main()