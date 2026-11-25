# Description: This script loads stratified training/test sets, builds a random forest baseline,
# performs RFECV feature selection with Accuracy and Macro-F1 scoring,
# outputs dual-score curves and feature-count-vs-score CSV files,
# and plots a combined Top-20 feature importance figure.
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder
import warnings
FONT_SCHEME = 'times'
FONT_FAMILIES = {
    'zh': ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'],
    'arial': ['Arial'],
    'times': ['Times New Roman'],
    'custom': ['SimHei', 'Microsoft YaHei', 'Arial']
}
font_list = FONT_FAMILIES.get(FONT_SCHEME, FONT_FAMILIES['times'])
plt.rcParams['font.sans-serif'] = font_list
plt.rcParams['axes.unicode_minus'] = False
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = font_list
rcParams['axes.unicode_minus'] = False
print("当前字体:", rcParams['font.sans-serif'])
warnings.filterwarnings('ignore')
print("LULC分类 - RFE特征选择（Accuracy & Macro-F1 双评分）")
def load_and_validate_data(train_path, test_path):
    print("步骤1: 数据加载与验证")
    print(f" 训练集路径: {os.path.abspath(train_path)}")
    print(f" 测试集路径: {os.path.abspath(test_path)}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"训练集大小: {train_df.shape}")
    print(f"测试集大小: {test_df.shape}")
    train_columns = set(train_df.columns)
    test_columns = set(test_df.columns)
    if train_columns != test_columns:
        print("错误: 训练集和测试集特征列不一致!")
        print(f"训练集特有列: {train_columns - test_columns}")
        print(f"测试集特有列: {test_columns - train_columns}")
        raise ValueError("训练集和测试集特征列必须完全一致")
    print("训练集和测试集特征列一致性验证通过")
    return train_df, test_df
train_df, test_df = load_and_validate_data(
    'training_set_stratified.csv',
    'test_set_stratified.csv'
)
print("\n 步骤2: 数据预处理")
non_feature_columns = ['landcover', 'longitude', 'latitude']
feature_columns = [col for col in train_df.columns if col not in non_feature_columns]
print(f"特征数量: {len(feature_columns)}")
print(f"前10个特征: {feature_columns[:10]}")
X_train = train_df[feature_columns]
y_train = train_df['landcover']
X_test = test_df[feature_columns]
y_test = test_df['landcover']
def handle_missing_data(X_train, X_test):
    missing_train = X_train.isnull().sum().sum()
    missing_test = X_test.isnull().sum().sum()
    print(f"训练集缺失值总数: {missing_train}")
    print(f"测试集缺失值总数: {missing_test}")
    if missing_train > 0 or missing_test > 0:
        print("正在进行缺失值填充...")
        train_means = X_train.mean()
        X_train_filled = X_train.fillna(train_means)
        X_test_filled = X_test.fillna(train_means)
        return X_train_filled, X_test_filled
    else:
        return X_train, X_test
X_train, X_test = handle_missing_data(X_train, X_test)
def encode_labels(y_train, y_test):
    if y_train.dtype == 'object' or y_test.dtype == 'object':
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        print("标签编码映射:")
        for i, class_name in enumerate(label_encoder.classes_):
            print(f"  {class_name} -> {i}")
        return y_train_encoded, y_test_encoded, label_encoder
    else:
        print("标签已为数值格式")
        return y_train.values, y_test.values, None
y_train_encoded, y_test_encoded, label_encoder = encode_labels(y_train, y_test)
def standardize_features(X_train, X_test, feature_columns):
    print("当前未进行特征标准化，直接使用原始特征。")
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    scaler = None
    return X_train_scaled, X_test_scaled, scaler
X_train_scaled, X_test_scaled, feature_scaler = standardize_features(X_train, X_test, feature_columns)
print("\n步骤3: 建立基准模型（全特征，用于特征重要性排序）")
baseline_model = RandomForestClassifier(
    n_estimators=200,
    max_features='sqrt',
    min_samples_leaf=1,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    verbose=0
)
baseline_model.fit(X_train_scaled, y_train_encoded)
print(f"基准模型训练完成，使用特征数量: {len(feature_columns)}")
print("\n步骤4: 递归特征消除 (RFE) - 双评分")
rfecv_acc = RFECV(
    estimator=RandomForestClassifier(
        n_estimators=200,
        max_features='sqrt',
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    ),
    step=1,
    cv=5,
    scoring='accuracy',
    min_features_to_select=1,
    n_jobs=-1,
    verbose=1
)
print("开始 RFECV（scoring='accuracy'）...")
rfecv_acc.fit(X_train_scaled, y_train_encoded)
print("RFECV (Accuracy) 完成!")
print(f"Accuracy 评分下的最优特征数量: {rfecv_acc.n_features_}")
rfecv_f1 = RFECV(
    estimator=RandomForestClassifier(
        n_estimators=200,
        max_features='sqrt',
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    ),
    step=1,
    cv=5,
    scoring='f1_macro',
    min_features_to_select=1,
    n_jobs=-1,
    verbose=1
)
print("\n开始 RFECV（scoring='f1_macro'）...")
rfecv_f1.fit(X_train_scaled, y_train_encoded)
print("RFECV (Macro-F1) 完成")
print(f"Macro-F1 评分下的最优特征数量: {rfecv_f1.n_features_}")
print("\n步骤5: 生成特征数量与评分对应表（Accuracy & Macro-F1）")
feature_numbers_acc = range(1, len(rfecv_acc.cv_results_['mean_test_score']) + 1)
mean_scores_acc = rfecv_acc.cv_results_['mean_test_score']
std_scores_acc = rfecv_acc.cv_results_['std_test_score']
cv_results_acc_df = pd.DataFrame({
    '特征数量': feature_numbers_acc,
    '平均Accuracy': mean_scores_acc,
    'Accuracy标准差': std_scores_acc,
    '是否最优(Accuracy)': [n_features == rfecv_acc.n_features_ for n_features in feature_numbers_acc]
})
cv_results_acc_df['Accuracy范围下限'] = cv_results_acc_df['平均Accuracy'] - cv_results_acc_df['Accuracy标准差']
cv_results_acc_df['Accuracy范围上限'] = cv_results_acc_df['平均Accuracy'] + cv_results_acc_df['Accuracy标准差']
cv_results_acc_df = cv_results_acc_df.sort_values('特征数量', ascending=False).reset_index(drop=True)
print("RFECV (Accuracy) 关键结果:")
print(f"最优特征数量 (Accuracy): {rfecv_acc.n_features_}")
print(f"最高 Accuracy: {cv_results_acc_df['平均Accuracy'].max():.4f}")
cv_results_acc_df.to_csv('rfecv_feature_vs_accuracy.csv', index=False, encoding='utf-8-sig')
print("Accuracy 曲线结果已保存: rfecv_feature_vs_accuracy.csv")
feature_numbers_f1 = range(1, len(rfecv_f1.cv_results_['mean_test_score']) + 1)
mean_scores_f1 = rfecv_f1.cv_results_['mean_test_score']
std_scores_f1 = rfecv_f1.cv_results_['std_test_score']
cv_results_f1_df = pd.DataFrame({
    '特征数量': feature_numbers_f1,
    '平均Macro-F1': mean_scores_f1,
    'Macro-F1标准差': std_scores_f1,
    '是否最优(Macro-F1)': [n_features == rfecv_f1.n_features_ for n_features in feature_numbers_f1]
})
cv_results_f1_df['Macro-F1范围下限'] = cv_results_f1_df['平均Macro-F1'] - cv_results_f1_df['Macro-F1标准差']
cv_results_f1_df['Macro-F1范围上限'] = cv_results_f1_df['平均Macro-F1'] + cv_results_f1_df['Macro-F1标准差']
cv_results_f1_df = cv_results_f1_df.sort_values('特征数量', ascending=False).reset_index(drop=True)
print("RFECV (Macro-F1) 关键结果:")
print(f"最优特征数量 (Macro-F1): {rfecv_f1.n_features_}")
print(f"最高 Macro-F1: {cv_results_f1_df['平均Macro-F1'].max():.4f}")
cv_results_f1_df.to_csv('rfecv_feature_vs_macro_f1.csv', index=False, encoding='utf-8-sig')
print("Macro-F1 曲线结果已保存: rfecv_feature_vs_macro_f1.csv")
print("\n步骤6: RFE结果分析（主视角: Macro-F1 选择）")
feature_analysis_df = pd.DataFrame({
    'Feature': feature_columns,
    'Selected': rfecv_f1.support_,
    'Ranking': rfecv_f1.ranking_
})
feature_analysis_df['Importance'] = baseline_model.feature_importances_
feature_analysis_df = feature_analysis_df.sort_values('Ranking')
selected_features_f1 = feature_analysis_df[feature_analysis_df['Selected']]['Feature'].tolist()
print(f"Macro-F1 评分下选中的特征数量: {len(selected_features_f1)}")
print(f"特征精简率 (Macro-F1): {(1 - len(selected_features_f1) / len(feature_columns)) * 100:.2f}%")
print("\nTop 10 最重要的特征 (按 Macro-F1 RFE 排名):")
top_features = feature_analysis_df.head(10)
for idx, row in top_features.iterrows():
    print(f"  {row['Ranking']:2d}. {row['Feature']:<20} (重要性: {row['Importance']:.4f})")
print("\n步骤7: 生成 RFE 组合图（左: 双评分曲线 / 右: Top20 特征重要性）")
FIG_SIZE = (14, 5.5)
LINE_WIDTH = 3
X_LABEL_LEFT = 'number'
Y_LABEL_LEFT = 'scores'
X_LABEL_RIGHT = 'Feature importance'
AXIS_LABEL_FONTSIZE_LEFT = 20
AXIS_LABEL_FONTSIZE_RIGHT = 20
TICK_LABEL_FONTSIZE_LEFT = 15
TICK_LABEL_FONTSIZE_RIGHT = 15
LEGEND_FONTSIZE = 13
BAR_YTICK_FONTSIZE = 12
GRID_LEFT = True
GRID_RIGHT = True
GRID_ALPHA = 0.2
TIGHT_LAYOUT_PAD = 3.0
TIGHT_LAYOUT_W_PAD = 3.0
base_name = 'RFE_Analysis_Results_sub12'
fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE)
ax1 = axes[0]
steps_acc = range(1, len(rfecv_acc.cv_results_['mean_test_score']) + 1)
steps_f1 = range(1, len(rfecv_f1.cv_results_['mean_test_score']) + 1)
ax1.plot(
    steps_acc,
    rfecv_acc.cv_results_['mean_test_score'],
    'b-',
    linewidth=LINE_WIDTH,
    label='Mean Accuracy (CV=5)'
)
ax1.plot(
    steps_f1,
    rfecv_f1.cv_results_['mean_test_score'],
    'g-',
    linewidth=LINE_WIDTH,
    label='Mean Macro-F1 (CV=5)'
)
ax1.axvline(
    x=rfecv_acc.n_features_,
    color='r',
    linestyle='--',
    label=f'features-Accuracy: {rfecv_acc.n_features_}'
)
ax1.axvline(
    x=rfecv_f1.n_features_,
    color='r',
    linestyle='--',
    label=f'features-Macro-F1: {rfecv_f1.n_features_}'
)
ax1.set_xlabel(X_LABEL_LEFT, fontsize=AXIS_LABEL_FONTSIZE_LEFT)
ax1.set_ylabel(Y_LABEL_LEFT, fontsize=AXIS_LABEL_FONTSIZE_LEFT)
ax1.legend(fontsize=LEGEND_FONTSIZE, loc='best')
ax1.grid(GRID_LEFT, alpha=GRID_ALPHA)
ax1.tick_params(labelsize=TICK_LABEL_FONTSIZE_LEFT)
ax2 = axes[1]
top_20_features = feature_analysis_df.head(20)
y_pos = range(len(top_20_features))
ax2.barh(y_pos, top_20_features['Importance'], height=0.8)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(top_20_features['Feature'], fontsize=BAR_YTICK_FONTSIZE)
ax2.set_xlabel(X_LABEL_RIGHT, fontsize=AXIS_LABEL_FONTSIZE_RIGHT)
ax2.invert_yaxis()
ax2.grid(GRID_RIGHT, alpha=GRID_ALPHA, axis='x')
ax2.tick_params(axis='x', labelsize=TICK_LABEL_FONTSIZE_RIGHT)
ax2.tick_params(axis='y', labelsize=BAR_YTICK_FONTSIZE)
plt.tight_layout(pad=TIGHT_LAYOUT_PAD, w_pad=TIGHT_LAYOUT_W_PAD)
fig.savefig(f'{base_name}.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"组合图已保存: {base_name}.png")
print(f"  • RFECV 双评分组合图:      {base_name}.png")
print("  • 特征数量-Accuracy 曲线表: rfecv_feature_vs_accuracy.csv")
print("  • 特征数量-Macro-F1 曲线表: rfecv_feature_vs_macro_f1.csv")
print(f"\n数据统计:")
print(f"  • 原始特征数量: {len(feature_columns)}")
print(f"  • Macro-F1 选中特征数量: {len(selected_features_f1)}")
print(f"  • Accuracy 选中特征数量: {int(rfecv_acc.support_.sum())}")
print(f"\n最重要的5个特征 (按 Macro-F1 RFE 排名):")
for rank, (_, row) in enumerate(feature_analysis_df.head(5).iterrows(), start=1):
    print(f"  {rank}. {row['Feature']} (重要性: {row['Importance']:.4f})")
print("=" * 70)