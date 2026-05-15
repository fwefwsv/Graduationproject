#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阿尔兹海默症智能分析预测系统 - Flask后端
基于XGBoost的机器学习预测API
来源：https://pythonhacker.blog.csdn.net/article/details/141941961
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import json
from datetime import datetime

# 创建Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'alzheimer-prediction-system-2024'

# 全局变量
model = None
feature_columns = None

def load_model():
    """加载训练好的模型和特征列"""
    global model, feature_columns
    try:
        model = joblib.load('alzheimer_model.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        print("模型加载成功！")
    except FileNotFoundError:
        print("警告：模型文件未找到，请先运行 model_training.py 训练模型")
        model = None
        feature_columns = None

@app.route('/')
def index():
    """系统首页"""
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    """预测页面"""
    return render_template('predict.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """预测API接口"""
    try:
        if model is None:
            load_model()
        if model is None:
            return jsonify({'error': '模型未加载，请先训练模型'}), 500

        # 获取输入数据
        data = request.json

        # 创建特征向量
        features = {}

        # 1. 特征映射表（英文 → 中文）
        feature_mapping = {
            'Age': '年龄',
            'Gender': '性别',
            'EducationLevel': '教育水平',
            'BMI': '身体质量指数（BMI）',
            'Smoking': '吸烟状况',
            'AlcoholConsumption': '酒精摄入量',
            'PhysicalActivity': '体育活动时间',
            'DietQuality': '饮食质量评分',
            'SleepQuality': '睡眠质量评分',
            'FamilyHistoryAlzheimers': '家族阿尔茨海默病史',
            'CardiovascularDisease': '心血管疾病',
            'Diabetes': '糖尿病',
            'Depression': '抑郁症史',
            'HeadInjury': '头部受伤',
            'Hypertension': '高血压',
            'SystolicBP': '收缩压',
            'DiastolicBP': '舒张压',
            'CholesterolTotal': '胆固醇总量',
            'CholesterolLDL': '低密度脂蛋白胆固醇（LDL）',
            'CholesterolHDL': '高密度脂蛋白胆固醇（HDL）',
            'CholesterolTriglycerides': '甘油三酯',
            'MMSE': '简易精神状态检查（MMSE）得分',
            'FunctionalAssessment': '功能评估得分',
            'MemoryComplaints': '记忆抱怨',
            'BehavioralProblems': '行为问题',
            'ADL': '日常生活活动（ADL）得分',
            'Confusion': '混乱与定向障碍',
            'Disorientation': '迷失方向',
            'PersonalityChanges': '人格变化',
            'DifficultyCompletingTasks': '完成任务困难',
            'Forgetfulness': '健忘'
        }

        # 填充基本特征：用英文key去data中取值，用中文key存入features
        for english_name, chinese_name in feature_mapping.items():
            value = data.get(english_name, 0)  # 用英文名获取数据
            features[chinese_name] = float(value)  # 用中文名存储

        # 2. 处理种族特征
        ethnicity_value = int(data.get('Ethnicity', 0))  # 前端传的是英文 'Ethnicity'

        # 添加种族one-hot编码特征（注意：训练时是 种族_1, 种族_2, 种族_3）
        features['种族_1'] = 1.0 if ethnicity_value == 1 else 0.0
        features['种族_2'] = 1.0 if ethnicity_value == 2 else 0.0
        features['种族_3'] = 1.0 if ethnicity_value == 3 else 0.0

        # 3. 创建DataFrame
        feature_df = pd.DataFrame([features])

        # 4. 确保所有训练时的特征列都存在
        print(f"训练时的特征列: {feature_columns}")
        print(f"当前特征列: {feature_df.columns.tolist()}")

        for col in feature_columns:
            if col not in feature_df.columns:
                print(f"警告：缺少特征列 {col}，设置为0")
                feature_df[col] = 0.0

        # 5. 按照训练时的特征顺序重新排列
        feature_df = feature_df[feature_columns]

        # 6. 创建DMatrix
        dmatrix = xgb.DMatrix(feature_df, feature_names=feature_columns)

        # 7. 预测
        prediction_proba = float(model.predict(dmatrix)[0])
        prediction = 1 if prediction_proba > 0.5 else 0

        # 计算风险等级
        if prediction_proba < 0.3:
            risk_level = "低风险"
            risk_color = "success"
        elif prediction_proba < 0.7:
            risk_level = "中等风险"
            risk_color = "warning"
        else:
            risk_level = "高风险"
            risk_color = "danger"

        # 获取特征重要性
        importance = model.get_score(importance_type='weight')
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]

        result = {
            'prediction': prediction,
            'probability': round(prediction_proba, 4),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'top_features': top_features
        }

        return jsonify(result)

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(error_msg)  # 服务器端打印详细错误
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics')
def get_statistics():
    """获取统计数据"""
    try:
        # 加载数据
        df = pd.read_csv("./alzheimers_disease_data.csv")
        # 年龄分布
        age_bins = pd.cut(df['Age'], bins=[60, 65, 70, 75, 80, 85, 90],
                          labels=['60-65', '66-70', '71-75', '76-80', '81-85', '86-90'])
        age_distribution = age_bins.value_counts().sort_index().to_dict()
        # 将numpy int64转为普通int，避免JSON序列化错误

        age_distribution = {str(k): int(v) for k, v in age_distribution.items()}
        # 基本统计
        stats = {
            'total_patients': len(df),
            'diagnosis_distribution': {
                'positive': int(df['Diagnosis'].sum()),
                'negative': int(len(df) - df['Diagnosis'].sum()),
                'positive_rate': round(df['Diagnosis'].mean() * 100, 2)
            },
            'age_statistics': {
                'mean': round(df['Age'].mean(), 1),
                'min': int(df['Age'].min()),
                'max': int(df['Age'].max())
            },
            'age_distribution': age_distribution,
            'gender_distribution': {
                'male': int((df['Gender'] == 0).sum()),
                'female': int((df['Gender'] == 1).sum())
            },
            'mmse_statistics': {
                'mean': round(df['MMSE'].mean(), 1),
                'min': int(df['MMSE'].min()),
                'max': int(df['MMSE'].max())
            }
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/features')
def get_features():
    """获取特征信息"""
    features = {
        'numeric_features': [
            {'name': '年龄', 'type': 'number', 'min': 60, 'max': 90, 'unit': '岁'},
            {'name': '性别', 'type': 'select', 'options': [{'value': 0, 'label': '男性'}, {'value': 1, 'label': '女性'}]},
            {'name': '种族', 'type': 'select', 'options': [
                {'value': 'Caucasian', 'label': '高加索人'},
                {'value': 'African American', 'label': '非裔美国人'},
                {'value': 'Asian', 'label': '亚洲人'},
                {'value': 'Other', 'label': '其他'}
            ]},
            {'name': '教育水平', 'type': 'number', 'min': 0, 'max': 20, 'unit': '年'},
            {'name': '身体质量指数（BMI）', 'type': 'number', 'min': 15, 'max': 40, 'unit': 'kg/m²'},
            {'name': '吸烟状况', 'type': 'select', 'options': [{'value': 0, 'label': '不吸烟'}, {'value': 1, 'label': '吸烟'}]},
            {'name': '酒精摄入量', 'type': 'number', 'min': 0, 'max': 20, 'unit': '单位/周'},
            {'name': '体育活动时间', 'type': 'number', 'min': 0, 'max': 10, 'unit': '小时/周'},
            {'name': '饮食质量评分', 'type': 'number', 'min': 0, 'max': 10, 'unit': '分'},
            {'name': '睡眠质量评分', 'type': 'number', 'min': 4, 'max': 10, 'unit': '分'},
            {'name': '家族阿尔茨海默病史', 'type': 'select', 'options': [{'value': 0, 'label': '无'}, {'value': 1, 'label': '有'}]},
            {'name': '心血管疾病', 'type': 'select', 'options': [{'value': 0, 'label': '无'}, {'value': 1, 'label': '有'}]},
            {'name': '糖尿病', 'type': 'select', 'options': [{'value': 0, 'label': '无'}, {'value': 1, 'label': '有'}]},
            {'name': '抑郁症史', 'type': 'select', 'options': [{'value': 0, 'label': '无'}, {'value': 1, 'label': '有'}]},
            {'name': '头部受伤', 'type': 'select', 'options': [{'value': 0, 'label': '无'}, {'value': 1, 'label': '有'}]},
            {'name': '高血压', 'type': 'select', 'options': [{'value': 0, 'label': '无'}, {'value': 1, 'label': '有'}]},
            {'name': '收缩压', 'type': 'number', 'min': 90, 'max': 180, 'unit': 'mmHg'},
            {'name': '舒张压', 'type': 'number', 'min': 60, 'max': 120, 'unit': 'mmHg'},
            {'name': '胆固醇总量', 'type': 'number', 'min': 150, 'max': 300, 'unit': 'mg/dL'},
            {'name': '低密度脂蛋白胆固醇（LDL）', 'type': 'number', 'min': 50, 'max': 200, 'unit': 'mg/dL'},
            {'name': '高密度脂蛋白胆固醇（HDL）', 'type': 'number', 'min': 20, 'max': 100, 'unit': 'mg/dL'},
            {'name': '甘油三酯', 'type': 'number', 'min': 50, 'max': 400, 'unit': 'mg/dL'},
            {'name': '简易精神状态检查（MMSE）得分', 'type': 'number', 'min': 0, 'max': 30, 'unit': '分'},
            {'name': '功能评估得分', 'type': 'number', 'min': 0, 'max': 10, 'unit': '分'},
            {'name': '记忆抱怨', 'type': 'select', 'options': [{'value': 0, 'label': '无'}, {'value': 1, 'label': '有'}]},
            {'name': '行为问题', 'type': 'select', 'options': [{'value': 0, 'label': '无'}, {'value': 1, 'label': '有'}]},
            {'name': '日常生活活动（ADL）得分', 'type': 'number', 'min': 0, 'max': 10, 'unit': '分'},
            {'name': '混乱与定向障碍', 'type': 'select', 'options': [{'value': 0, 'label': '无'}, {'value': 1, 'label': '有'}]},
            {'name': '迷失方向', 'type': 'select', 'options': [{'value': 0, 'label': '无'}, {'value': 1, 'label': '有'}]},
            {'name': '人格变化', 'type': 'select', 'options': [{'value': 0, 'label': '无'}, {'value': 1, 'label': '有'}]},
            {'name': '完成任务困难', 'type': 'select', 'options': [{'value': 0, 'label': '无'}, {'value': 1, 'label': '有'}]},
            {'name': '健忘', 'type': 'select', 'options': [{'value': 0, 'label': '无'}, {'value': 1, 'label': '有'}]}
        ]
    }
    
    return jsonify(features)

@app.route('/static/<path:filename>')
def static_files(filename):
    """提供静态文件"""
    return send_from_directory('static', filename)

@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    # 创建必要的目录
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    # 加载模型
    load_model()
    
    # 启动应用
    app.run(debug=True, host='0.0.0.0', port=5000)