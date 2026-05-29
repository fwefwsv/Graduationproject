#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost学习曲线绘制工具
参考: https://www.cnblogs.com/tingstone/p/17189088.html
"""

import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_learning_curve(evals_result, save_path='static/images/learning_curve.png'):
    """
    绘制XGBoost学习曲线
    
    参数:
        evals_result: XGBoost训练过程中记录的评估结果字典
                      通过xgb.train()的evals_result参数获取
        save_path: 图片保存路径，默认为 'static/images/learning_curve.png'
    
    返回:
        best_epoch: 最优迭代轮数（早停点）
    """
    print("正在绘制学习曲线...")
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 提取训练集和验证集的AUC历史
    train_auc = evals_result['train']['auc']
    valid_auc = evals_result['valid']['auc']
    epochs = range(len(train_auc))
    
    plt.figure(figsize=(10, 6))
    
    # 绘制训练集和验证集的学习曲线
    plt.plot(epochs, train_auc, 'b-', label='训练集 AUC', linewidth=2)
    plt.plot(epochs, valid_auc, 'orange', label='验证集 AUC', linewidth=2)
    
    # 标记最优迭代点（早停点）
    best_epoch = len(train_auc) - 1
    plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7, 
                label=f'早停点 (第{best_epoch}轮)')
    
    plt.xlabel('迭代轮数 (Boosting Round)', fontsize=12)
    plt.ylabel('AUC', fontsize=12)
    plt.title('XGBoost 学习曲线 (AUC)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"学习曲线已保存至: {save_path}")
    return best_epoch


def plot_learning_curve_full(evals_result, save_path='static/images/learning_curve_full.png'):
    """
    绘制完整的XGBoost学习曲线（包含多个评估指标）
    
    参数:
        evals_result: XGBoost训练过程中记录的评估结果字典
        save_path: 图片保存路径
    
    返回:
        best_epoch: 最优迭代轮数
    """
    print("正在绘制完整学习曲线...")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 绘制AUC曲线
    train_auc = evals_result['train'].get('auc', [])
    valid_auc = evals_result['valid'].get('auc', [])
    
    if train_auc:
        epochs = range(len(train_auc))
        axes[0].plot(epochs, train_auc, 'b-', label='训练集 AUC', linewidth=2)
        axes[0].plot(epochs, valid_auc, 'orange', label='验证集 AUC', linewidth=2)
        best_epoch = len(train_auc) - 1
        axes[0].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7,
                        label=f'早停点 (第{best_epoch}轮)')
        axes[0].set_xlabel('迭代轮数', fontsize=12)
        axes[0].set_ylabel('AUC', fontsize=12)
        axes[0].set_title('XGBoost AUC学习曲线', fontsize=14, fontweight='bold')
        axes[0].legend(loc='lower right', fontsize=11)
        axes[0].grid(True, alpha=0.3)
    
    # 绘制logloss曲线（如果存在）
    train_logloss = evals_result['train'].get('logloss', [])
    valid_logloss = evals_result['valid'].get('logloss', [])
    
    if train_logloss:
        epochs = range(len(train_logloss))
        axes[1].plot(epochs, train_logloss, 'b-', label='训练集 LogLoss', linewidth=2)
        axes[1].plot(epochs, valid_logloss, 'orange', label='验证集 LogLoss', linewidth=2)
        axes[1].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7,
                        label=f'早停点 (第{best_epoch}轮)')
        axes[1].set_xlabel('迭代轮数', fontsize=12)
        axes[1].set_ylabel('LogLoss', fontsize=12)
        axes[1].set_title('XGBoost LogLoss曲线', fontsize=14, fontweight='bold')
        axes[1].legend(loc='upper right', fontsize=11)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"完整学习曲线已保存至: {save_path}")
    return best_epoch


if __name__ == "__main__":
    print("XGBoost学习曲线绘制工具")
    print("使用方法：在XGBoost训练时传入evals_result参数")
    print("示例：")
    print("import xgboost as xgb")
    print("from xgboost_plot import plot_learning_curve")
    print()
    print("evals_result = {}")
    print("model = xgb.train(params, dtrain, num_boost_round=100,")
    print("                  evals=[(dtrain, 'train'), (dvalid, 'valid')],")
    print("                  evals_result=evals_result,")
    print("                  early_stopping_rounds=10)")
    print()
    print("plot_learning_curve(evals_result)")
