#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学习曲线诊断示意图 - 良好拟合/过拟合/欠拟合对比
参考: https://www.cnblogs.com/tingstone/p/17189088.html
"""

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_learning_curve_diagnosis(save_path='static/images/learning_curve_diagnosis.png'):
    """
    绘制学习曲线诊断示意图，展示三种拟合情况的对比
    
    参数:
        save_path: 图片保存路径
    
    返回:
        None
    """
    print("正在绘制学习曲线诊断示意图...")
    
    # 创建数据
    epochs = np.arange(1, 51)
    
    # 1. 良好拟合 (Good Fit)
    # 训练集和验证集误差都较低，且差距很小
    train_good = 0.8 - 0.6 * np.exp(-epochs / 5) + 0.02 * np.random.randn(len(epochs))
    valid_good = 0.82 - 0.55 * np.exp(-epochs / 6) + 0.02 * np.random.randn(len(epochs))
    
    # 2. 过拟合 (Overfitting)
    # 训练集误差很低，但验证集误差在后期上升
    train_over = 0.95 - 0.85 * np.exp(-epochs / 4) + 0.01 * np.random.randn(len(epochs))
    valid_over = 0.8 - 0.4 * np.exp(-epochs / 8) + 0.03 * np.random.randn(len(epochs))
    # 添加过拟合特征：验证集误差后期上升
    valid_over[20:] += 0.003 * (epochs[20:] - 20)
    
    # 3. 欠拟合 (Underfitting)
    # 训练集和验证集误差都很高，收敛在较高水平
    train_under = 0.6 - 0.2 * np.exp(-epochs / 10) + 0.02 * np.random.randn(len(epochs))
    valid_under = 0.62 - 0.18 * np.exp(-epochs / 12) + 0.02 * np.random.randn(len(epochs))
    
    # 确保值在合理范围内
    train_good = np.clip(train_good, 0.15, 0.95)
    valid_good = np.clip(valid_good, 0.15, 0.95)
    train_over = np.clip(train_over, 0.05, 0.95)
    valid_over = np.clip(valid_over, 0.15, 0.95)
    train_under = np.clip(train_under, 0.3, 0.95)
    valid_under = np.clip(valid_under, 0.3, 0.95)
    
    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 良好拟合
    axes[0].plot(epochs, train_good, 'b-', label='训练集', linewidth=2)
    axes[0].plot(epochs, valid_good, 'orange', label='验证集', linewidth=2)
    axes[0].fill_between(epochs, train_good, valid_good, where=(train_good > valid_good), 
                         color='lightblue', alpha=0.3)
    axes[0].fill_between(epochs, train_good, valid_good, where=(train_good < valid_good), 
                         color='orange', alpha=0.1)
    axes[0].set_xlabel('训练轮数', fontsize=12)
    axes[0].set_ylabel('AUC', fontsize=12)
    axes[0].set_title('(a) 良好拟合', fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0.1, 1.0)
    
    # 添加诊断说明
    axes[0].text(30, 0.92, '✓ 训练集与验证集\nAUC均较高', fontsize=10, 
                 bbox=dict(facecolor='green', alpha=0.2))
    
    # 过拟合
    axes[1].plot(epochs, train_over, 'b-', label='训练集', linewidth=2)
    axes[1].plot(epochs, valid_over, 'orange', label='验证集', linewidth=2)
    axes[1].fill_between(epochs, train_over, valid_over, color='red', alpha=0.1)
    axes[1].axvline(x=20, color='r', linestyle='--', alpha=0.7, label='开始过拟合')
    axes[1].set_xlabel('训练轮数', fontsize=12)
    axes[1].set_ylabel('AUC', fontsize=12)
    axes[1].set_title('(b) 过拟合', fontsize=14, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.1, 1.0)
    
    # 添加诊断说明
    axes[1].text(30, 0.92, '✗ 训练集AUC很高\n✗ 验证集AUC下降', fontsize=10, 
                 bbox=dict(facecolor='red', alpha=0.2))
    
    # 欠拟合
    axes[2].plot(epochs, train_under, 'b-', label='训练集', linewidth=2)
    axes[2].plot(epochs, valid_under, 'orange', label='验证集', linewidth=2)
    axes[2].fill_between(epochs, train_under, valid_under, color='yellow', alpha=0.1)
    axes[2].set_xlabel('训练轮数', fontsize=12)
    axes[2].set_ylabel('AUC', fontsize=12)
    axes[2].set_title('(c) 欠拟合', fontsize=14, fontweight='bold')
    axes[2].legend(loc='lower right', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0.1, 1.0)
    
    # 添加诊断说明
    axes[2].text(30, 0.92, '✗ 训练集与验证集\nAUC均较低', fontsize=10, 
                 bbox=dict(facecolor='yellow', alpha=0.3))
    
    # 整体标题
    fig.suptitle('XGBoost学习曲线诊断示意图', fontsize=16, fontweight='bold', y=1.02)
    
    # 添加底部说明
    fig.text(0.5, -0.08, 
             '诊断准则：训练集与验证集差距小且AUC高→良好拟合；训练集AUC高但验证集下降→过拟合；两者AUC均低→欠拟合',
             ha='center', fontsize=11, color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"学习曲线诊断示意图已保存至: {save_path}")


def plot_learning_curve_analysis(save_path='static/images/learning_curve_analysis.png'):
    """
    绘制学习曲线分析图，包含诊断建议
    """
    print("正在绘制学习曲线分析图...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左侧：典型学习曲线
    epochs = np.arange(1, 31)
    train_auc = 0.75 + 0.2 * (1 - np.exp(-epochs / 5)) + 0.01 * np.random.randn(len(epochs))
    valid_auc = 0.72 + 0.18 * (1 - np.exp(-epochs / 6)) + 0.015 * np.random.randn(len(epochs))
    
    ax1.plot(epochs, train_auc, 'b-', label='训练集 AUC', linewidth=2)
    ax1.plot(epochs, valid_auc, 'orange', label='验证集 AUC', linewidth=2)
    
    # 标记关键区域
    ax1.axvspan(0, 8, color='green', alpha=0.1, label='欠拟合区域')
    ax1.axvspan(8, 20, color='blue', alpha=0.1, label='良好拟合区域')
    ax1.axvspan(20, 30, color='red', alpha=0.1, label='过拟合区域')
    
    ax1.axvline(x=8, color='green', linestyle='--', alpha=0.5)
    ax1.axvline(x=20, color='red', linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('训练轮数', fontsize=12)
    ax1.set_ylabel('AUC', fontsize=12)
    ax1.set_title('典型学习曲线与拟合状态', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.6, 1.0)
    
    # 右侧：诊断决策树
    decision_tree = """
诊断决策流程
─────────────

              开始训练
                  │
                  ▼
      训练集AUC是否 > 0.85？
          /           \\
        是             否
         │              │
         ▼              ▼
   验证集AUC是否   训练集欠拟合
   > 0.80？        增加模型复杂度
       / \\
     是   否
      │    │
      ▼    ▼
   良好拟合  过拟合
   继续训练  早停/正则化
    """
    
    ax2.text(0.1, 0.1, decision_tree, fontsize=11, family='monospace', 
             bbox=dict(facecolor='lightgray', alpha=0.5), transform=ax2.transAxes)
    ax2.set_title('学习曲线诊断决策流程', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"学习曲线分析图已保存至: {save_path}")


if __name__ == "__main__":
    # 生成诊断示意图
    plot_learning_curve_diagnosis()
    plot_learning_curve_analysis()
    print("\n学习曲线诊断示意图生成完成！")
