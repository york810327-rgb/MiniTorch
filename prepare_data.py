import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False
# 加载数据
X_train = pd.read_csv('X_train.txt', sep=r'\s+', header=None)
y_train = pd.read_csv('y_train.txt', sep=r'\s+', header=None)
X_test = pd.read_csv('X_test.txt', sep=r'\s+', header=None)
y_test = pd.read_csv('y_test.txt', sep=r'\s+', header=None)
y_train = y_train - 1
y_test = y_test - 1
# 活动标签映射 
activity_labels = {
    0: '步行 (Walk)', 
    1: '上楼 (Upstairs)', 
    2: '下楼 (Downstairs)', 
    3: '坐 (Sit)', 
    4: '站 (Stand)', 
    5: '躺 (Lay)'
}
# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 模型训练与评估
print("真实数据加载成功！")
print("训练集 X 形状:", X_train.shape)
print("测试集 X 形状:", X_test.shape)
# 标签处理
y_train_adjusted = y_train - 1
y_test_adjusted = y_test - 1
# 配置并初始化 LightGBM 模型
print("开始训练 LightGBM 模型，请稍候...")
clf = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=6,          
    random_state=42,      
    n_estimators=100      
)
# 把数据喂给模型进行训练
clf.fit(X_train, y_train_adjusted.values.ravel())
# 让模型在测试集上做预测
y_pred = clf.predict(X_test)
# 准确率计算
acc = accuracy_score(y_test_adjusted, y_pred)
print(f"模型训练完成！测试集准确率: {acc * 100:.2f}%")
# 打印结果
print("\n 详细分类报告:")
print(classification_report(y_test_adjusted, y_pred, target_names=[
    '步行 (Walk)', '上楼 (Upstairs)', '下楼 (Downstairs)', 
    '坐着 (Sitting)', '站立 (Standing)', '躺着 (Laying)'
]))