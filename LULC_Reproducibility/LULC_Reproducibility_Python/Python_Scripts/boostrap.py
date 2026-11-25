# Description: This file reads CSV files of reference/predicted labels exported from GEE, computes confusion matrices,
# OA, Kappa, Macro-F1, and per-class PA/UA/F1, and uses stratified bootstrap to estimate 95% confidence intervals,
# outputting result tables for Shanxi and Shaanxi respectively.

import numpy as np
import pandas as pd
INPUT_CSV = "LULC_Test_Predictions_for_Bootstrap.csv"
OUTPUT_CSV = "LULC_Test_Predictions_for_Bootstrap_results.csv"
LABEL_COL = "landcover"
PRED_COL  = "classification"
N_BOOT = 1000
RANDOM_SEED = 42
def load_data(csv_path, label_col, pred_col):
    df = pd.read_csv(csv_path)
    if label_col not in df.columns or pred_col not in df.columns:
        raise ValueError(f"CSV 中必须包含列: {label_col!r} 和 {pred_col!r}")
    y_true = df[label_col].values
    y_pred = df[pred_col].values
    classes = np.unique(y_true)
    classes = np.sort(classes)
    return y_true, y_pred, classes
def confusion_and_metrics(y_true, y_pred, classes):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.asarray(classes)
    C = len(classes)
    N = len(y_true)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((C, C), dtype=int)
    for t, p in zip(y_true, y_pred):
        i = class_to_idx[t]
        j = class_to_idx[p]
        cm[i, j] += 1
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    correct = np.trace(cm)
    oa = correct / N if N > 0 else np.nan
    pe = np.sum(row_sum * col_sum) / (N ** 2) if N > 0 else np.nan
    if N > 0 and (1 - pe) != 0:
        kappa = (oa - pe) / (1 - pe)
    else:
        kappa = np.nan
    tp = np.diag(cm).astype(float)
    pa = np.divide(tp, row_sum, out=np.zeros_like(tp), where=row_sum > 0)
    ua = np.divide(tp, col_sum, out=np.zeros_like(tp), where=col_sum > 0)
    f1 = np.zeros(C, dtype=float)
    for i in range(C):
        if pa[i] + ua[i] > 0:
            f1[i] = 2 * pa[i] * ua[i] / (pa[i] + ua[i])
        else:
            f1[i] = 0.0
    macro_f1 = np.nanmean(f1) if C > 0 else np.nan
    return cm, oa, kappa, pa, ua, f1, macro_f1
def stratified_bootstrap(y_true, y_pred, classes, n_boot=1000, random_seed=42):
    rng = np.random.default_rng(random_seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.asarray(classes)
    C = len(classes)
    N = len(y_true)
    indices_by_class = {
        c: np.where(y_true == c)[0]
        for c in classes
    }
    for c, idxs in indices_by_class.items():
        if len(idxs) == 0:
            raise ValueError(f"类别 {c} 在测试集中没有样本，无法做分层 bootstrap。")
    oa_arr = np.zeros(n_boot)
    kappa_arr = np.zeros(n_boot)
    macro_f1_arr = np.zeros(n_boot)
    pa_arr = np.zeros((n_boot, C))
    ua_arr = np.zeros((n_boot, C))
    f1_arr = np.zeros((n_boot, C))
    for b in range(n_boot):
        boot_indices_list = []
        for c in classes:
            idxs = indices_by_class[c]
            n_c = len(idxs)
            boot_c = rng.choice(idxs, size=n_c, replace=True)
            boot_indices_list.append(boot_c)
        boot_indices = np.concatenate(boot_indices_list)
        y_true_bs = y_true[boot_indices]
        y_pred_bs = y_pred[boot_indices]
        _, oa, kappa, pa, ua, f1, macro_f1 = confusion_and_metrics(
            y_true_bs, y_pred_bs, classes
        )
        oa_arr[b] = oa
        kappa_arr[b] = kappa
        macro_f1_arr[b] = macro_f1
        pa_arr[b, :] = pa
        ua_arr[b, :] = ua
        f1_arr[b, :] = f1
    return oa_arr, kappa_arr, macro_f1_arr, pa_arr, ua_arr, f1_arr
def main():
    y_true, y_pred, classes = load_data(INPUT_CSV, LABEL_COL, PRED_COL)
    C = len(classes)
    N = len(y_true)
    print(f"样本数 N = {N}, 类别数 C = {C}, 类别标签 = {classes.tolist()}")
    print("\n计算一次性混淆矩阵与指标（点估计）...")
    cm, oa, kappa, pa, ua, f1, macro_f1 = confusion_and_metrics(y_true, y_pred, classes)
    support = cm.sum(axis=1)
    print("混淆矩阵：")
    print(pd.DataFrame(cm, index=[f"true_{c}" for c in classes],
                          columns=[f"pred_{c}" for c in classes]))
    print(f"\nOA       = {oa:.4f}")
    print(f"Kappa    = {kappa:.4f}")
    print(f"Macro-F1 = {macro_f1:.4f}")
    print("\n每类指标（点估计）：")
    df_per_class_point = pd.DataFrame({
        "class": classes,
        "support": support,
        "PA": pa,
        "UA": ua,
        "F1": f1
    })
    print(df_per_class_point)
    print("\n开始类别分层 bootstrap ...")
    print(f"bootstrap 次数: {N_BOOT}, 随机种子: {RANDOM_SEED}")
    (oa_arr, kappa_arr, macro_f1_arr,
     pa_arr, ua_arr, f1_arr) = stratified_bootstrap(
        y_true, y_pred, classes, n_boot=N_BOOT, random_seed=RANDOM_SEED
    )
    def ci_95(arr):
        return np.quantile(arr, [0.025, 0.975])
    oa_ci = ci_95(oa_arr)
    kappa_ci = ci_95(kappa_arr)
    mf1_ci = ci_95(macro_f1_arr)
    print("\nBootstrap 95% 置信区间：")
    print(f"OA:       {oa:.4f}  (95% CI: {oa_ci[0]:.4f} ~ {oa_ci[1]:.4f})")
    print(f"Kappa:    {kappa:.4f}  (95% CI: {kappa_ci[0]:.4f} ~ {kappa_ci[1]:.4f})")
    print(f"Macro-F1: {macro_f1:.4f}  (95% CI: {mf1_ci[0]:.4f} ~ {mf1_ci[1]:.4f})")
    pa_ci = np.quantile(pa_arr, [0.025, 0.975], axis=0)
    ua_ci = np.quantile(ua_arr, [0.025, 0.975], axis=0)
    f1_ci = np.quantile(f1_arr, [0.025, 0.975], axis=0)
    overall_rows = [
        {
            "level": "overall",
            "metric": "OA",
            "estimate": oa,
            "ci_lower": oa_ci[0],
            "ci_upper": oa_ci[1],
            "n_boot": N_BOOT,
            "n_test": N
        },
        {
            "level": "overall",
            "metric": "Kappa",
            "estimate": kappa,
            "ci_lower": kappa_ci[0],
            "ci_upper": kappa_ci[1],
            "n_boot": N_BOOT,
            "n_test": N
        },
        {
            "level": "overall",
            "metric": "Macro-F1",
            "estimate": macro_f1,
            "ci_lower": mf1_ci[0],
            "ci_upper": mf1_ci[1],
            "n_boot": N_BOOT,
            "n_test": N
        }
    ]
    df_overall = pd.DataFrame(overall_rows)
    per_class_rows = []
    for i, cls in enumerate(classes):
        per_class_rows.append({
            "level": "per_class",
            "class": cls,
            "support": int(support[i]),
            "PA": pa[i],
            "PA_ci_lower": pa_ci[0, i],
            "PA_ci_upper": pa_ci[1, i],
            "UA": ua[i],
            "UA_ci_lower": ua_ci[0, i],
            "UA_ci_upper": ua_ci[1, i],
            "F1": f1[i],
            "F1_ci_lower": f1_ci[0, i],
            "F1_ci_upper": f1_ci[1, i],
            "n_boot": N_BOOT,
            "n_test": N
        })
    df_per_class = pd.DataFrame(per_class_rows)
    df_S1 = pd.concat([df_overall, df_per_class], ignore_index=True)
    df_S1.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
INPUT_CSV = "Shaanxi_all_for_Bootstrap.csv"
OUTPUT_CSV = "Shaanxi_all_for_Bootstrap_results.csv"
LABEL_COL = "landcover"
PRED_COL  = "classification"
N_BOOT = 1000
RANDOM_SEED = 42
def load_data(csv_path, label_col, pred_col):
    df = pd.read_csv(csv_path)
    if label_col not in df.columns or pred_col not in df.columns:
        raise ValueError(f"CSV 中必须包含列: {label_col!r} 和 {pred_col!r}")
    y_true = df[label_col].values
    y_pred = df[pred_col].values
    classes = np.unique(y_true)
    classes = np.sort(classes)
    return y_true, y_pred, classes
def confusion_and_metrics(y_true, y_pred, classes):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.asarray(classes)
    C = len(classes)
    N = len(y_true)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((C, C), dtype=int)
    for t, p in zip(y_true, y_pred):
        i = class_to_idx[t]
        j = class_to_idx[p]
        cm[i, j] += 1
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    correct = np.trace(cm)
    oa = correct / N if N > 0 else np.nan
    pe = np.sum(row_sum * col_sum) / (N ** 2) if N > 0 else np.nan
    if N > 0 and (1 - pe) != 0:
        kappa = (oa - pe) / (1 - pe)
    else:
        kappa = np.nan
    tp = np.diag(cm).astype(float)
    pa = np.divide(tp, row_sum, out=np.zeros_like(tp), where=row_sum > 0)
    ua = np.divide(tp, col_sum, out=np.zeros_like(tp), where=col_sum > 0)
    f1 = np.zeros(C, dtype=float)
    for i in range(C):
        if pa[i] + ua[i] > 0:
            f1[i] = 2 * pa[i] * ua[i] / (pa[i] + ua[i])
        else:
            f1[i] = 0.0
    macro_f1 = np.nanmean(f1) if C > 0 else np.nan
    return cm, oa, kappa, pa, ua, f1, macro_f1
def stratified_bootstrap(y_true, y_pred, classes, n_boot=1000, random_seed=42):
    rng = np.random.default_rng(random_seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.asarray(classes)
    C = len(classes)
    N = len(y_true)
    indices_by_class = {
        c: np.where(y_true == c)[0]
        for c in classes
    }
    for c, idxs in indices_by_class.items():
        if len(idxs) == 0:
            raise ValueError(f"类别 {c} 在测试集中没有样本，无法做分层 bootstrap。")
    oa_arr = np.zeros(n_boot)
    kappa_arr = np.zeros(n_boot)
    macro_f1_arr = np.zeros(n_boot)
    pa_arr = np.zeros((n_boot, C))
    ua_arr = np.zeros((n_boot, C))
    f1_arr = np.zeros((n_boot, C))
    for b in range(n_boot):
        boot_indices_list = []
        for c in classes:
            idxs = indices_by_class[c]
            n_c = len(idxs)
            boot_c = rng.choice(idxs, size=n_c, replace=True)
            boot_indices_list.append(boot_c)
        boot_indices = np.concatenate(boot_indices_list)
        y_true_bs = y_true[boot_indices]
        y_pred_bs = y_pred[boot_indices]
        _, oa, kappa, pa, ua, f1, macro_f1 = confusion_and_metrics(
            y_true_bs, y_pred_bs, classes
        )
        oa_arr[b] = oa
        kappa_arr[b] = kappa
        macro_f1_arr[b] = macro_f1
        pa_arr[b, :] = pa
        ua_arr[b, :] = ua
        f1_arr[b, :] = f1
    return oa_arr, kappa_arr, macro_f1_arr, pa_arr, ua_arr, f1_arr
def main():
    y_true, y_pred, classes = load_data(INPUT_CSV, LABEL_COL, PRED_COL)
    C = len(classes)
    N = len(y_true)
    print(f"样本数 N = {N}, 类别数 C = {C}, 类别标签 = {classes.tolist()}")
    print("\n计算一次性混淆矩阵与指标（点估计）...")
    cm, oa, kappa, pa, ua, f1, macro_f1 = confusion_and_metrics(y_true, y_pred, classes)
    support = cm.sum(axis=1)
    print("混淆矩阵：")
    print(pd.DataFrame(cm, index=[f"true_{c}" for c in classes],
                          columns=[f"pred_{c}" for c in classes]))
    print(f"\nOA       = {oa:.4f}")
    print(f"Kappa    = {kappa:.4f}")
    print(f"Macro-F1 = {macro_f1:.4f}")
    print("\n每类指标（点估计）：")
    df_per_class_point = pd.DataFrame({
        "class": classes,
        "support": support,
        "PA": pa,
        "UA": ua,
        "F1": f1
    })
    print(df_per_class_point)
    print("\n开始类别分层 bootstrap ...")
    print(f"bootstrap 次数: {N_BOOT}, 随机种子: {RANDOM_SEED}")
    (oa_arr, kappa_arr, macro_f1_arr,
     pa_arr, ua_arr, f1_arr) = stratified_bootstrap(
        y_true, y_pred, classes, n_boot=N_BOOT, random_seed=RANDOM_SEED
    )
    def ci_95(arr):
        return np.quantile(arr, [0.025, 0.975])
    oa_ci = ci_95(oa_arr)
    kappa_ci = ci_95(kappa_arr)
    mf1_ci = ci_95(macro_f1_arr)
    print("\nBootstrap 95% 置信区间：")
    print(f"OA:       {oa:.4f}  (95% CI: {oa_ci[0]:.4f} ~ {oa_ci[1]:.4f})")
    print(f"Kappa:    {kappa:.4f}  (95% CI: {kappa_ci[0]:.4f} ~ {kappa_ci[1]:.4f})")
    print(f"Macro-F1: {macro_f1:.4f}  (95% CI: {mf1_ci[0]:.4f} ~ {mf1_ci[1]:.4f})")
    pa_ci = np.quantile(pa_arr, [0.025, 0.975], axis=0)
    ua_ci = np.quantile(ua_arr, [0.025, 0.975], axis=0)
    f1_ci = np.quantile(f1_arr, [0.025, 0.975], axis=0)
    overall_rows = [
        {
            "level": "overall",
            "metric": "OA",
            "estimate": oa,
            "ci_lower": oa_ci[0],
            "ci_upper": oa_ci[1],
            "n_boot": N_BOOT,
            "n_test": N
        },
        {
            "level": "overall",
            "metric": "Kappa",
            "estimate": kappa,
            "ci_lower": kappa_ci[0],
            "ci_upper": kappa_ci[1],
            "n_boot": N_BOOT,
            "n_test": N
        },
        {
            "level": "overall",
            "metric": "Macro-F1",
            "estimate": macro_f1,
            "ci_lower": mf1_ci[0],
            "ci_upper": mf1_ci[1],
            "n_boot": N_BOOT,
            "n_test": N
        }
    ]
    df_overall = pd.DataFrame(overall_rows)
    per_class_rows = []
    for i, cls in enumerate(classes):
        per_class_rows.append({
            "level": "per_class",
            "class": cls,
            "support": int(support[i]),
            "PA": pa[i],
            "PA_ci_lower": pa_ci[0, i],
            "PA_ci_upper": pa_ci[1, i],
            "UA": ua[i],
            "UA_ci_lower": ua_ci[0, i],
            "UA_ci_upper": ua_ci[1, i],
            "F1": f1[i],
            "F1_ci_lower": f1_ci[0, i],
            "F1_ci_upper": f1_ci[1, i],
            "n_boot": N_BOOT,
            "n_test": N
        })
    df_per_class = pd.DataFrame(per_class_rows)
    df_S1 = pd.concat([df_overall, df_per_class], ignore_index=True)
    df_S1.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
if __name__ == "__main__":
    main()