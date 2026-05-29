#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost学习曲线绘制工具
用于可视化模型训练过程中的学习曲线
参考: https://www.cnblogs.com/tingstone/p/17189088.html
"""

import matplotlib.pyplot as plt
import os

# 设置中文字体（与项目保持一致）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_learning_curve(evals_result, save_path='static/images/learning_curve.png'):
    """
    绘制XGBoost学习曲线
    
    参数:
        evals_result: XGBoost训练过程中记录的评估结果字典
                      格式: {'train': {'auc': [values]}, 'valid': {'auc': [values]}}
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


def plot_learning_curve_with_loss(evals_result, save_path='static/images/learning_curve_full.png'):
    """
    绘制包含损失函数的完整学习曲线
    
    参数:
        evals_result: XGBoost训练过程中记录的评估结果字典
        save_path: 图片保存路径
    
    返回:
        best_epoch: 最优迭代轮数
    """
    print("正在绘制完整学习曲线（包含损失）...")
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 绘制AUC曲线
    train_auc = evals_result['train'].get('auc', [])
    valid_auc = evals_result['valid'].get('auc', [])
    if train_auc:
        epochs = range(len(train_auc))
        ax1.plot(epochs, train_auc, 'b-', label='训练集 AUC', linewidth=2)
        ax1.plot(epochs, valid_auc, 'orange', label='验证集 AUC', linewidth=2)
        best_epoch = len(train_auc) - 1
        ax1.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7,
                    label=f'早停点 (第{best_epoch}轮)')
        ax1.set_xlabel('迭代轮数', fontsize=12)
        ax1.set_ylabel('AUC', fontsize=12)
        ax1.set_title('XGBoost AUC学习曲线', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=11)
        ax1.grid(True, alpha=0.3)
    
    # 绘制损失曲线
    train_error = evals_result['train'].get('error', [])
    valid_error = evals_result['valid'].get('error', [])
    if train_error:
        epochs = range(len(train_error))
        ax2.plot(epochs, train_error, 'b-', label='训练集 Error', linewidth=2)
        ax2.plot(epochs, valid_error, 'orange', label='验证集 Error', linewidth=2)
        ax2.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7,
                    label=f'早停点 (第{best_epoch}轮)')
        ax2.set_xlabel('迭代轮数', fontsize=12)
        ax2.set_ylabel('Error', fontsize=12)
        ax2.set_title('XGBoost 损失曲线', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=11)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"完整学习曲线已保存至: {save_path}")
    return best_epoch


if __name__ == "__main__":
    # 示例用法（需要配合XGBoost训练使用）
    # 1. 首先需要训练模型并获取evals_result
    # 示例代码：
    # import xgboost as xgb
    # evals_result = {}
    # model = xgb.train(params, dtrain, num_boost_round=100,
    #                   evals=[(dtrain, 'train'), (dvalid, 'valid')],
    #                   evals_result=evals_result,
    #                   early_stopping_rounds=10)
    # 
    # 2. 然后调用绘制函数
    # plot_learning_curve(evals_result)
    
    print("注意：此文件需要在XGBoost训练过程中调用，")
    print("请在model_training.py中训练模型时传入evals_result参数。")
