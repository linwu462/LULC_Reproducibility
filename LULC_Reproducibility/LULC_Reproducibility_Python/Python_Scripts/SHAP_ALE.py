# Description: This script reads the slim training/test sets, trains a random forest, computes SHAP values,
# and outputs four figures: a class-proportion stacked SHAP mean bar chart, a 6-class SHAP beeswarm grid,
# a 6-class × Top-5-feature ALE grid, and a 6-class × Top-5-feature PDP/ICE grid.
import os
import warnings
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")
OUTDIR = "shap_outputs"
os.makedirs(OUTDIR, exist_ok=True)
TOPK_PER_CLASS = 5
FONT_SCHEME = "times"
FONT_FAMILY_MAP = {
    "default": ["DejaVu Sans"],
    "msyh": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "arial": ["Arial", "DejaVu Sans"],
    "times": ["Times New Roman", "DejaVu Sans"],
}
font_families = FONT_FAMILY_MAP.get(FONT_SCHEME, ["DejaVu Sans"])
plt.rcParams["font.family"] = font_families
plt.rcParams["axes.unicode_minus"] = False
PDP_ICE_XLABEL_FONTSIZE = 20
PDP_ICE_YLABEL_FONTSIZE = 20
PDP_ICE_TICK_LABEL_SIZE = 18
PDP_ICE_CLASS_LABEL_FONTSIZE = 16
PDP_ICE_RANK_LABEL_FONTSIZE = 14
PDP_ICE_SAMPLE_LABEL_FONTSIZE = 15
PDP_ICE_LEGEND_CONFIG = {
    "loc": "upper center",
    "bbox_to_anchor": (0.5, 0.02),
    "ncol": None,
    "fontsize": 16,
    "frameon": True,
    "fancybox": True,
    "shadow": True,
    "framealpha": 0.9,
}
ALE_GRID_XLABEL_FONTSIZE = 16
ALE_GRID_YLABEL_FONTSIZE = 15
ALE_GRID_TICK_LABELSIZE = 14
ALE_GRID_LEGEND_CONFIG = {
    "loc": "upper center",
    "bbox_to_anchor": (0.5, 0.02),
    "ncol": None,
    "fontsize": 15,
    "frameon": True,
    "fancybox": True,
    "shadow": False,
    "framealpha": 0.9,
}
STACKED_SHAP_FIG_WIDTH = 10.0
STACKED_SHAP_ROW_HEIGHT = 0.25
STACKED_SHAP_MIN_HEIGHT = 6.0
STACKED_SHAP_XTICK_FONTSIZE = 13
STACKED_SHAP_YTICK_FONTSIZE = 13
STACKED_SHAP_XLABEL_FONTSIZE = 15
STACKED_SHAP_LEGEND_MODE = "inside_lower_right"
STACKED_SHAP_LEGEND_FONTSIZE = 13
STACKED_SHAP_LEGEND_ANCHORS = {
    "outside_right": (1.02, 1.0),
    "outside_right_lower": (1.02, 0.5),
    "inside_lower_right": (1.0, 0.0),
}
STACKED_SHAP_SPINES_VISIBLE = {
    "left": True,
    "right": False,
    "top": False,
    "bottom": True,
}
BEESWARM_GRID_N_COLS = 3
BEESWARM_GRID_BASE_WIDTH_PER_COL = 6.0
BEESWARM_GRID_BASE_HEIGHT_PER_ROW = 4.0
BEESWARM_GRID_EXTRA_HEIGHT = 4.0
BEESWARM_GRID_LEFT = 0.05
BEESWARM_GRID_RIGHT = 0.95
BEESWARM_GRID_BOTTOM = 0.08
BEESWARM_GRID_TOP = 0.95
BEESWARM_GRID_WSPACE = 0.015
BEESWARM_GRID_HSPACE = 0.07
BEESWARM_INNER_FIGSIZE = (6.0, 6.0)
BEESWARM_INNER_TICK_LABELSIZE = 16
BEESWARM_INNER_AXIS_LABELSIZE = 18
BEESWARM_INNER_TITLE_FONTSIZE = 18
def compute_ale_1d_class(model, X, feature, class_idx, n_bins=20):
    X_local = X.copy().reset_index(drop=True)
    x = X_local[feature].to_numpy()
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.quantile(x, quantiles)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        raise ValueError(f"特征 {feature} 变异度过低，无法计算 ALE。")
    bin_edges[0] -= 1e-8
    bin_edges[-1] += 1e-8
    K = len(bin_edges) - 1
    ale_increments = np.zeros(K, dtype=float)
    for k in range(K):
        lower = bin_edges[k]
        upper = bin_edges[k + 1]
        mask = (x > lower) & (x <= upper)
        if not np.any(mask):
            ale_increments[k] = 0.0
            continue
        X_low = X_local.copy()
        X_high = X_local.copy()
        X_low.loc[mask, feature] = lower
        X_high.loc[mask, feature] = upper
        proba_low = model.predict_proba(X_low)
        proba_high = model.predict_proba(X_high)
        f_low = proba_low[mask, class_idx]
        f_high = proba_high[mask, class_idx]
        ale_increments[k] = np.mean(f_high - f_low)
    ale_values = np.cumsum(ale_increments)
    ale_values -= np.mean(ale_values)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, ale_values, bin_edges
train_path = "slim_training19.csv"
test_path = "slim_test19.csv"
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
possible_label_cols = ["landcover", "label", "class", "LULC"]
label_col = [c for c in train_df.columns if c.lower() in possible_label_cols][0]
possible_lon = ["lon", "longitude"]
possible_lat = ["lat", "latitude"]
lon_col = next((c for c in train_df.columns if c.lower() in possible_lon), None)
lat_col = next((c for c in train_df.columns if c.lower() in possible_lat), None)
print(f"自动识别：label_col = {label_col}  lon_col = {lon_col}  lat_col = {lat_col}")
drop_cols = [label_col]
if lon_col:
    drop_cols.append(lon_col)
if lat_col:
    drop_cols.append(lat_col)
X_train = train_df.drop(columns=drop_cols)
y_train = train_df[label_col].astype(str)
X_test = test_df.drop(columns=drop_cols)
y_test = test_df[label_col].astype(str)
print("训练样本数：", len(X_train), " 测试样本数：", len(X_test))
print("特征数量：", X_train.shape[1])
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
scaler = None
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)
print("类别映射：", dict(zip(le.transform(le.classes_), le.classes_)))
baseline_model = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",
    min_samples_leaf=1,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    verbose=0,
)
baseline_model.fit(X_train_scaled, y_train_enc)
print("开始计算 SHAP 值...")
explainer = shap.TreeExplainer(baseline_model)
raw_shap_values = explainer.shap_values(X_test_scaled)
shap_arr = np.array(raw_shap_values)
if isinstance(raw_shap_values, list):
    shap_arr = np.transpose(shap_arr, (1, 2, 0))
elif shap_arr.ndim == 3 and shap_arr.shape[0] == X_test_scaled.shape[0] and shap_arr.shape[2] == len(
    le.classes_
):
    pass
elif shap_arr.ndim == 3 and shap_arr.shape[0] == X_test_scaled.shape[0] and shap_arr.shape[1] == len(
    le.classes_
):
    shap_arr = np.transpose(shap_arr, (0, 2, 1))
else:
    raise ValueError(f"无法识别的 SHAP 形状: {shap_arr.shape}")
shap_values = shap_arr
n_samples, n_features, n_classes = shap_values.shape
features = X_test_scaled.columns.tolist()
class_names = list(le.classes_)
print(f"SHAP 值形状: {shap_values.shape}")
mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 2))
print("正在生成 SHAP 蜂群图...")
_bee_old_xtick = plt.rcParams.get("xtick.labelsize", 10)
_bee_old_ytick = plt.rcParams.get("ytick.labelsize", 10)
_bee_old_axes_label = plt.rcParams.get("axes.labelsize", 12)
plt.rcParams["xtick.labelsize"] = BEESWARM_INNER_TICK_LABELSIZE
plt.rcParams["ytick.labelsize"] = BEESWARM_INNER_TICK_LABELSIZE
plt.rcParams["axes.labelsize"] = BEESWARM_INNER_AXIS_LABELSIZE
n_cols = BEESWARM_GRID_N_COLS
n_rows = int(np.ceil(n_classes / n_cols))
temp_files = []
for cls_idx, cls_name in enumerate(class_names):
    fig = plt.figure(figsize=BEESWARM_INNER_FIGSIZE)
    shap.summary_plot(
        shap_values[:, :, cls_idx],
        X_test_scaled,
        show=False,
        plot_size=None,
    )
    for ax in fig.get_axes():
        if ax.get_label() == "<colorbar>":
            ax.remove()
            break
    plt.title(
        f"Class {cls_name}",
        fontsize=BEESWARM_INNER_TITLE_FONTSIZE,
        fontweight="bold",
        pad=10,
    )
    plt.xlabel("SHAP value", fontsize=BEESWARM_INNER_AXIS_LABELSIZE)
    plt.tight_layout()
    temp_file = os.path.join(OUTDIR, f"temp_class_{cls_name}.png")
    plt.savefig(temp_file, dpi=600, bbox_inches="tight")
    plt.close()
    temp_files.append(temp_file)
fig_width = BEESWARM_GRID_BASE_WIDTH_PER_COL * n_cols
fig_height = BEESWARM_GRID_BASE_HEIGHT_PER_ROW * n_rows + BEESWARM_GRID_EXTRA_HEIGHT
fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
axes_flat = axes.flatten()
for idx, (temp_file, cls_name) in enumerate(zip(temp_files, class_names)):
    img = plt.imread(temp_file)
    axes_flat[idx].imshow(img)
    axes_flat[idx].axis("off")
for idx in range(n_classes, len(axes_flat)):
    axes_flat[idx].set_visible(False)
plt.subplots_adjust(
    left=BEESWARM_GRID_LEFT,
    right=BEESWARM_GRID_RIGHT,
    bottom=BEESWARM_GRID_BOTTOM,
    top=BEESWARM_GRID_TOP,
    wspace=BEESWARM_GRID_WSPACE,
    hspace=BEESWARM_GRID_HSPACE,
)
combined_beeswarm = os.path.join(OUTDIR, "shap_summary_dot_all_classes.png")
plt.savefig(combined_beeswarm, dpi=600, bbox_inches="tight")
plt.close()
for temp_file in temp_files:
    if os.path.exists(temp_file):
        os.remove(temp_file)
print("组合SHAP散点分布图已保存：", combined_beeswarm)
plt.rcParams["xtick.labelsize"] = _bee_old_xtick
plt.rcParams["ytick.labelsize"] = _bee_old_ytick
plt.rcParams["axes.labelsize"] = _bee_old_axes_label
print("\n开始生成按类别比例堆叠的 SHAP 均值图...")
M_fc = np.mean(np.abs(shap_values), axis=0)
mean_abs_shap_vec = mean_abs_shap
sum_Mc = M_fc.sum(axis=1, keepdims=True)
with np.errstate(divide="ignore", invalid="ignore"):
    P_fc = np.where(sum_Mc == 0, 0, M_fc / sum_Mc)
H_fc = mean_abs_shap_vec.reshape(-1, 1) * P_fc
df_segments = pd.DataFrame(H_fc, columns=class_names, index=features)
df_totals = pd.DataFrame({"Feature": features, "Mean|SHAP|": mean_abs_shap_vec}).set_index(
    "Feature"
)
order = df_totals["Mean|SHAP|"].sort_values(ascending=False).index
df_segments = df_segments.loc[order]
df_totals = df_totals.loc[order]
fig_height = max(STACKED_SHAP_MIN_HEIGHT, STACKED_SHAP_ROW_HEIGHT * len(df_segments))
fig, ax = plt.subplots(figsize=(STACKED_SHAP_FIG_WIDTH, fig_height))
y_pos = np.arange(len(df_segments))
left = np.zeros(len(df_segments))
color_tab = plt.cm.tab10.colors
for i_cls, cls in enumerate(class_names):
    widths = df_segments[cls].values
    ax.barh(
        y_pos,
        widths,
        left=left,
        height=0.8,
        label=cls,
        color=color_tab[i_cls % len(color_tab)],
    )
    left += widths
ax.set_yticks(y_pos)
ax.set_yticklabels(df_segments.index, fontsize=STACKED_SHAP_YTICK_FONTSIZE)
ax.invert_yaxis()
ax.set_xlabel(
    "Mean(|SHAP value|) (total = global Mean(|SHAP|)",
    fontsize=STACKED_SHAP_XLABEL_FONTSIZE,
)
ax.tick_params(axis="x", labelsize=STACKED_SHAP_XTICK_FONTSIZE)
for spine_pos, visible in STACKED_SHAP_SPINES_VISIBLE.items():
    ax.spines[spine_pos].set_visible(visible)
legend_mode = STACKED_SHAP_LEGEND_MODE
anchor = STACKED_SHAP_LEGEND_ANCHORS.get(
    legend_mode, STACKED_SHAP_LEGEND_ANCHORS["outside_right"]
)
if legend_mode == "inside_lower_right":
    legend_loc = "lower right"
elif legend_mode == "outside_right_lower":
    legend_loc = "lower left"
else:
    legend_loc = "upper left"
ax.legend(
    title="Class",
    loc=legend_loc,
    bbox_to_anchor=anchor,
    fontsize=13,
)
plt.tight_layout()
stack_path = os.path.join(OUTDIR, "shap_mean_importance_stacked.png")
plt.savefig(stack_path, dpi=600)
plt.close()
print("堆叠 SHAP 图已保存：", stack_path)
print("计算“按地类 Top-5 特征”的 ALE 曲线（与 PDP/ICE 对应）...")
X_ale = pd.concat([X_train_scaled, X_test_scaled], axis=0)
ale_class_results = {}
df_abs_importance_cls = pd.DataFrame(M_fc, columns=class_names, index=features)
ale_class_dir = os.path.join(OUTDIR, "ale_curves_by_class")
os.makedirs(ale_class_dir, exist_ok=True)
for cls_idx, cls_name in enumerate(class_names):
    top_features_cls = (
        df_abs_importance_cls[cls_name]
        .abs()
        .sort_values(ascending=False)
        .head(TOPK_PER_CLASS)
        .index
        .tolist()
    )
    print(f"\n地类 {cls_name} 的 Top-{TOPK_PER_CLASS} 特征用于 ALE：", top_features_cls)
    for feat in top_features_cls:
        if feat not in X_ale.columns:
            print(f"  特征 {feat} 不在 X_ale 中，跳过。")
            continue
        print(f"  计算地类 {cls_name} – 特征 {feat} 的 ALE 曲线...")
        try:
            bin_centers, ale_values, bin_edges = compute_ale_1d_class(
                baseline_model, X_ale, feat, class_idx=cls_idx, n_bins=20
            )
        except Exception as e:
            print(f"   计算地类 {cls_name} – 特征 {feat} 的 ALE 时出错：{e}")
            continue
        values = X_ale[feat].values
        p5, p95 = np.percentile(values, [5, 95])
        y_min, y_max = ale_values.min(), ale_values.max()
        margin = 0.05 * (y_max - y_min + 1e-8)
        y_min_plot = y_min - margin
        y_max_plot = y_max + margin
        ale_class_results[(cls_name, feat)] = (bin_centers, ale_values)
print(f"按地类 Top-{TOPK_PER_CLASS} 特征的 ALE 计算完成。")
print("生成 ALE 组图（按地类 Top-5 特征）...")
legend_fontsize = 15
legend_ncol = 2
legend_handlelength = 2.0
legend_handletextpad = 0.8
legend_columnspacing = 1.5
legend_labelspacing = 0.6
legend_borderpad = 0.6
legend_y = 0.02
bottom_margin = 0.06
n_rows_group = len(class_names)
n_cols_group = TOPK_PER_CLASS
fig, axes = plt.subplots(
    n_rows_group,
    n_cols_group,
    figsize=(n_cols_group * 3.2, n_rows_group * 2.5),
    squeeze=False,
)
for row_idx, cls_name in enumerate(class_names):
    top_features_cls = (
        df_abs_importance_cls[cls_name]
        .abs()
        .sort_values(ascending=False)
        .head(TOPK_PER_CLASS)
        .index
        .tolist()
    )
    for col_idx in range(n_cols_group):
        ax = axes[row_idx, col_idx]
        if col_idx >= len(top_features_cls):
            ax.set_visible(False)
            continue
        feat = top_features_cls[col_idx]
        key = (cls_name, feat)
        if key not in ale_class_results:
            ax.text(
                0.5,
                0.5,
                "No ALE\nresult",
                ha="center",
                va="center",
                fontsize=8,
                color="red",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            continue
        x_vals, y_vals = ale_class_results[key]
        values_all = X_ale[feat].values
        p5, p95 = np.percentile(values_all, [5, 95])
        y_min, y_max = y_vals.min(), y_vals.max()
        margin = 0.05 * (y_max - y_min + 1e-8)
        y_min_plot = y_min - margin
        y_max_plot = y_max + margin
        ax.plot(x_vals, y_vals, linewidth=2)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.fill_betweenx(
            [y_min_plot, y_max_plot],
            p5,
            p95,
            color="gray",
            alpha=0.1,
        )
        ax.axvline(p5, color="gray", linestyle=":", linewidth=1)
        ax.axvline(p95, color="gray", linestyle=":", linewidth=1)
        ax.set_ylim(y_min_plot, y_max_plot)
        ax.set_xlabel(feat, fontsize=ALE_GRID_XLABEL_FONTSIZE)
        if col_idx == 0:
            ax.set_ylabel(
                "ALE effect on\nP(class={})".format(cls_name),
                fontsize=ALE_GRID_YLABEL_FONTSIZE,
            )
            ax.tick_params(axis="y", labelleft=True, labelsize=ALE_GRID_TICK_LABELSIZE)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False, labelsize=ALE_GRID_TICK_LABELSIZE)
        ax.set_title("")
        ax.tick_params(axis="x", labelsize=ALE_GRID_TICK_LABELSIZE)
legend_handles = [
    Line2D([0], [0], color="tab:blue", lw=2, label="ALE curve"),
    Line2D(
        [0],
        [0],
        color="gray",
        lw=0,
        marker="s",
        markersize=10,
        markerfacecolor="lightgray",
        alpha=0.3,
        label="5–95% quantile range",
    ),
]
legend = fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, legend_y),
    ncol=legend_ncol,
    fontsize=legend_fontsize,
    handlelength=legend_handlelength,
    handletextpad=legend_handletextpad,
    columnspacing=legend_columnspacing,
    labelspacing=legend_labelspacing,
    borderpad=legend_borderpad,
    frameon=True,
    fancybox=True,
)
plt.subplots_adjust(
    left=0.10,
    right=0.98,
    top=0.96,
    bottom=bottom_margin,
    wspace=0.25,
    hspace=0.35,
)
ale_grid_path = os.path.join(ale_class_dir, f"ale_by_class_top{TOPK_PER_CLASS}_grid.png")
plt.savefig(ale_grid_path, dpi=600, bbox_inches="tight")
plt.close(fig)
print(f" ALE 组图已保存：{ale_grid_path}")
print(" 开始生成综合 PDP/ICE 图（灰色ICE线版本）...")
df_abs_importance = pd.DataFrame(M_fc, columns=class_names, index=features)
print("\n📏 计算全局特征取值范围并预定义网格点...")
feature_grids = {}
grid_resolution = 40
for feature in features:
    min_val = X_test[feature].min()
    max_val = X_test[feature].max()
    grid_points = np.linspace(min_val, max_val, grid_resolution)
    feature_grids[feature] = grid_points
    print(f"  {feature}: [{min_val:.3f}, {max_val:.3f}], 网格点: {len(grid_points)}")
def manual_pdp_ice_calculation(model, X_data, feature, class_idx, grid_points):
    n_samples_local = X_data.shape[0]
    n_grid = len(grid_points)
    ice_values = np.zeros((n_samples_local, n_grid))
    for i, grid_val in enumerate(grid_points):
        X_temp = X_data.copy()
        X_temp[feature] = grid_val
        pred_proba = model.predict_proba(X_temp)
        ice_values[:, i] = pred_proba[:, class_idx]
    pdp_values = np.mean(ice_values, axis=0)
    return pdp_values, ice_values, grid_points
os.makedirs("final_comprehensive_gray_ice", exist_ok=True)
topk = TOPK_PER_CLASS
print("\n 生成综合图...")
n_classes_local = len(class_names)
fig, axes = plt.subplots(
    n_classes_local, topk, figsize=(topk * 4, n_classes_local * 3.5), squeeze=False
)
base_colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
class_colors = {
    cls_name: base_colors[i % len(base_colors)] for i, cls_name in enumerate(class_names)
}
plt.rcParams["axes.labelsize"] = PDP_ICE_XLABEL_FONTSIZE
plt.rcParams["axes.titlesize"] = PDP_ICE_RANK_LABEL_FONTSIZE
plt.rcParams["xtick.labelsize"] = PDP_ICE_TICK_LABEL_SIZE
plt.rcParams["ytick.labelsize"] = PDP_ICE_TICK_LABEL_SIZE
for i_cls, cls_name in enumerate(class_names):
    print(f"处理地类 [{cls_name}] ...")
    class_mask = y_test_enc == i_cls
    X_test_class = X_test.loc[class_mask]
    n_samples_class = len(X_test_class)
    if n_samples_class == 0:
        for j in range(topk):
            ax = axes[i_cls, j]
            ax.text(
                0.5,
                0.5,
                f"No samples\nfor class {cls_name}",
                ha="center",
                va="center",
                fontsize=9,
                color="red",
            )
            ax.set_axis_off()
        continue
    top_features = (
        df_abs_importance[cls_name]
        .abs()
        .sort_values(ascending=False)
        .head(topk)
        .index
        .tolist()
    )
    for j, feat in enumerate(top_features):
        ax = axes[i_cls, j]
        try:
            grid_points = feature_grids[feat]
            pdp_values, ice_values, _ = manual_pdp_ice_calculation(
                baseline_model, X_test_class, feat, i_cls, grid_points
            )
            for ice_line in ice_values:
                ax.plot(grid_points, ice_line, color="gray", alpha=0.2, linewidth=0.6)
            ax.plot(
                grid_points,
                pdp_values,
                color=class_colors[cls_name],
                linewidth=3,
                alpha=0.9,
                label=f"{cls_name} (n={n_samples_class})",
            )
            min_val = X_test[feat].min()
            max_val = X_test[feat].max()
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(0, 1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_alpha(0.3)
            ax.spines["bottom"].set_alpha(0.3)
            ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
            ax.set_xlabel(feat, fontsize=PDP_ICE_XLABEL_FONTSIZE, labelpad=8)
            if j == 0:
                ax.set_ylabel(
                    "Predicted\nProbability",
                    fontsize=PDP_ICE_YLABEL_FONTSIZE,
                    labelpad=8,
                )
            else:
                ax.set_ylabel("")
            ax.tick_params(
                axis="x",
                which="both",
                bottom=True,
                labelbottom=True,
                length=3,
                labelsize=PDP_ICE_TICK_LABEL_SIZE,
            )
            ax.tick_params(
                axis="y",
                which="both",
                left=True,
                labelleft=(j == 0),
                length=3,
                labelsize=PDP_ICE_TICK_LABEL_SIZE,
            )
            if j == 0:
                ax.text(
                    -0.25,
                    0.98,
                    cls_name,
                    transform=ax.transAxes,
                    fontsize=PDP_ICE_CLASS_LABEL_FONTSIZE,
                    fontweight="bold",
                    va="top",
                    ha="right",
                    color=class_colors[cls_name],
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        alpha=0.8,
                        edgecolor=class_colors[cls_name],
                    ),
                )
            ax.text(
                0.98,
                0.02,
                f"n={n_samples_class}",
                transform=ax.transAxes,
                fontsize=PDP_ICE_SAMPLE_LABEL_FONTSIZE,
                va="bottom",
                ha="right",
                alpha=0.7,
                bbox=dict(
                    boxstyle="round,pad=0.1",
                    facecolor="white",
                    alpha=0.6,
                    edgecolor="none",
                ),
            )
            ax.text(
                0.98,
                0.98,
                f"Rank {j + 1}",
                transform=ax.transAxes,
                fontsize=PDP_ICE_RANK_LABEL_FONTSIZE,
                fontweight="bold",
                va="top",
                ha="right",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="lightyellow",
                    alpha=0.8,
                    edgecolor="gray",
                ),
            )
        except Exception as e:
            ax.clear()
            ax.text(
                0.5,
                0.5,
                f"Error\n{feat}",
                ha="center",
                va="center",
                fontsize=9,
                color="red",
                alpha=0.7,
            )
            ax.set_facecolor("#f8f8f8")
            ax.set_xlabel(feat, fontsize=PDP_ICE_XLABEL_FONTSIZE, labelpad=8)
            ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True)
            ax.tick_params(
                axis="y",
                which="both",
                left=True,
                labelleft=(j == 0),
                labelsize=PDP_ICE_TICK_LABEL_SIZE,
            )
            print(f"⚠绘图失败 - class {cls_name}, feat {feat}: {e}")
legend_elems = []
for cls_name in class_names:
    legend_elems.append(
        Line2D(
            [0],
            [0],
            color=class_colors[cls_name],
            lw=3,
            label=f"{cls_name}",
            alpha=0.9,
        )
    )
legend_elems.extend(
    [
        Line2D([0], [0], color="gray", lw=1, alpha=0.4, label="ICE Lines (Gray)"),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="lightyellow",
            markersize=8,
            label="Feature Rank",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="white",
            markersize=6,
            label="Sample Size",
        ),
    ]
)
legend_kwargs = PDP_ICE_LEGEND_CONFIG.copy()
if legend_kwargs.get("ncol") is None:
    legend_kwargs["ncol"] = len(legend_elems)
fig.legend(handles=legend_elems, **legend_kwargs)
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
comprehensive_path = "final_comprehensive_gray_ice/comprehensive_pdp_ice_gray_ice.png"
plt.savefig(
    comprehensive_path,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
plt.close(fig)
print(f" 综合图（灰色ICE线版本）已保存：{comprehensive_path}")
print("完成，共输出 4 张图：")
print(f"  1) {stack_path}")
print(f"  2) {combined_beeswarm}")
print(f"  3) {ale_grid_path}")
print(f"  4) {comprehensive_path}")