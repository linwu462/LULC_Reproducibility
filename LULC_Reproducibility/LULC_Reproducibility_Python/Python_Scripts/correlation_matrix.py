# Description: Read 19 RFECV-selected features, compute the Pearson correlation matrix,
# and plot a lower-triangle dot-style correlation matrix.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
FONT_SCHEME = 'times'
FONT_FAMILIES = {
    'msyh': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'arial': ['Arial', 'DejaVu Sans'],
    'times': ['Times New Roman', 'DejaVu Sans'],
    'default': None
}
font_family = FONT_FAMILIES.get(FONT_SCHEME, None)
if font_family is not None:
    plt.rcParams['font.family'] = font_family
    plt.rcParams['axes.unicode_minus'] = False
FIG_SIZE = (10, 8)
TITLE_TEXT = "Correlation matrix of RFECV-selected 19 features"
TITLE_FONTSIZE = 20
TITLE_FONTWEIGHT = 'bold'
X_LABEL_TEXT = ""
Y_LABEL_TEXT = ""
DEFAULT_LABEL_SIZE = plt.rcParams.get('axes.labelsize', 18)
AXIS_LABEL_FONTSIZE = DEFAULT_LABEL_SIZE
AXIS_LABEL_FONTWEIGHT = 'bold'
TICK_LABEL_FONTSIZE_X = 16
TICK_LABEL_FONTSIZE_Y = 16
XTICK_ROTATION = 45
XTICK_HA = 'right'
YTICK_ROTATION = 0
COLORBAR_LABEL_TEXT = ""
COLORBAR_LABEL_FONTSIZE = 15
COLORBAR_TICK_FONTSIZE = 13
COLORBAR_TICK_LENGTH = 4
COLORBAR_TICK_WIDTH = 0.8
DOT_SIZE_MIN = 50
DOT_SIZE_MAX = 400
INPUT_CSV = "slim_training19.csv"
OUTPUT_PNG = "corr_matrix_19features_circle_only.png"
def load_training_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df
def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7",
        "NDVI", "EVI", "MSAVI",
        "MNDWI", "NDTI", "BLFEI", "DBSI", "MBI",
        "slope",
        "MNDWI_contrast", "MNDWI_idm", "SWIR1_asm",
    ]
    X = df[feature_cols].copy()
    X = X.astype(float)
    return X
def compute_corr_matrix(X: pd.DataFrame) -> pd.DataFrame:
    corr = X.corr(method="pearson")
    return corr
def plot_corr_heatmap(corr: pd.DataFrame, save_path: str = OUTPUT_PNG) -> None:
    n_features = corr.shape[0]
    feature_names = corr.columns.tolist()
    corr_values = corr.values
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    xs, ys, vals = [], [], []
    for i in range(n_features):
        for j in range(n_features):
            if i >= j:
                xs.append(j)
                ys.append(i)
                vals.append(corr_values[i, j])
    xs = np.array(xs)
    ys = np.array(ys)
    vals = np.array(vals)
    max_abs = np.nanmax(np.abs(vals))
    if max_abs == 0:
        max_abs = 1.0
    sizes = DOT_SIZE_MIN + (DOT_SIZE_MAX - DOT_SIZE_MIN) * (np.abs(vals) / max_abs)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=-1, vmax=1)
    sc = ax.scatter(xs, ys, c=vals, s=sizes, cmap=cmap, norm=norm)
    ax.set_xlim(-0.5, n_features - 0.5)
    ax.set_ylim(n_features - 0.5, -0.5)
    ax.set_xticks(np.arange(n_features))
    ax.set_yticks(np.arange(n_features))
    ax.set_xticklabels(
        feature_names,
        rotation=XTICK_ROTATION,
        ha=XTICK_HA,
        fontsize=TICK_LABEL_FONTSIZE_X
    )
    ax.set_yticklabels(
        feature_names,
        rotation=YTICK_ROTATION,
        fontsize=TICK_LABEL_FONTSIZE_Y
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        TITLE_TEXT,
        fontsize=TITLE_FONTSIZE,
        fontweight=TITLE_FONTWEIGHT
    )
    if X_LABEL_TEXT:
        ax.set_xlabel(
            X_LABEL_TEXT,
            fontsize=AXIS_LABEL_FONTSIZE,
            fontweight=AXIS_LABEL_FONTWEIGHT
        )
    if Y_LABEL_TEXT:
        ax.set_ylabel(
            Y_LABEL_TEXT,
            fontsize=AXIS_LABEL_FONTSIZE,
            fontweight=AXIS_LABEL_FONTWEIGHT
        )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(
        COLORBAR_LABEL_TEXT,
        fontsize=COLORBAR_LABEL_FONTSIZE
    )
    cbar.ax.tick_params(
        labelsize=COLORBAR_TICK_FONTSIZE,
        length=COLORBAR_TICK_LENGTH,
        width=COLORBAR_TICK_WIDTH
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
def main():
    df = load_training_data(INPUT_CSV)
    X = get_feature_matrix(df)
    corr = compute_corr_matrix(X)
    plot_corr_heatmap(corr, save_path=OUTPUT_PNG)
    print(f"相关矩阵圆点图已保存：{OUTPUT_PNG}")
if __name__ == "__main__":
    main()