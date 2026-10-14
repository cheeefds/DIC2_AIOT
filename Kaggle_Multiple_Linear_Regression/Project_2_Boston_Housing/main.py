"""
Project 2: Boston Housing Price Prediction using Multiple Linear Regression
Executing Machine Learning Workflow & 5-Method Feature Selection Evaluation
"""

import os
import json
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.feature_selection import SequentialFeatureSelector, RFE
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Styling configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def load_and_inspect_data(csv_path: str):
    """Load Boston Housing dataset, standardize column casing, and inspect statistics."""
    print("=" * 70)
    print("STEP 1: DATASET LOADING AND INSPECTION")
    print("=" * 70)
    df = pd.read_csv(csv_path)
    # Standardize column names to uppercase
    df.columns = [c.upper() for c in df.columns]
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nSummary Statistics:")
    print(df.describe().round(2))
    return df


def exploratory_data_analysis(df: pd.DataFrame, output_dir: str):
    """Compute correlation matrix and save high-resolution heatmap."""
    print("\n" + "=" * 70)
    print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)
    os.makedirs(output_dir, exist_ok=True)
    corr = df.corr()
    print("Top correlated features with MEDV:")
    print(corr['MEDV'].sort_values(ascending=False).round(4))
    
    plt.figure(figsize=(11, 9), dpi=300)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.8, linecolor="white",
                cbar_kws={"shrink": 0.8}, annot_kws={"size": 9, "weight": "bold"})
    plt.title("Boston Housing: Correlation Heatmap (13 Predictors & MEDV)", fontsize=14, weight='bold', pad=15)
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Saved correlation heatmap: {heatmap_path}")


def evaluate_features_cv(X_tr: np.ndarray, y_tr: np.ndarray, 
                         feature_indices: list, splits: list):
    """Evaluate linear regression using standard scaler strictly on train folds."""
    rmse_scores = []
    r2_scores = []
    
    for train_idx, val_idx in splits:
        X_fold_tr = X_tr[train_idx][:, feature_indices]
        y_fold_tr = y_tr[train_idx]
        X_fold_va = X_tr[val_idx][:, feature_indices]
        y_fold_va = y_tr[val_idx]
        
        scaler = StandardScaler()
        X_fold_tr_sc = scaler.fit_transform(X_fold_tr)
        X_fold_va_sc = scaler.transform(X_fold_va)
        
        model = LinearRegression()
        model.fit(X_fold_tr_sc, y_fold_tr)
        preds = model.predict(X_fold_va_sc)
        
        rmse_scores.append(root_mean_squared_error(y_fold_va, preds))
        r2_scores.append(r2_score(y_fold_va, preds))
        
    return np.mean(rmse_scores), np.mean(r2_scores)


def run_feature_selection(X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series):
    """
    Execute 5 feature selection methods over 1 to 13 features:
    1. Lasso (L1 Ranking)
    2. Forward SFS
    3. RFE
    4. ElasticNet Ranking
    5. Exhaustive Search
    """
    print("\n" + "=" * 70)
    print("STEP 3: FEATURE SELECTION COMPARISON ACROSS 5 METHODS (k=1..13)")
    print("=" * 70)
    
    feature_names = X_train.columns.tolist()
    num_features_total = len(feature_names)
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(kf.split(X_train))
    
    X_train_np = X_train.values
    y_train_np = y_train.values
    X_test_np = X_test.values
    y_test_np = y_test.values
    
    scaler_full = StandardScaler()
    X_train_sc = scaler_full.fit_transform(X_train)
    
    # -------------------------------------------------------------
    # 1. Lasso (L1 Ranking)
    # -------------------------------------------------------------
    print("Running Method 1: Lasso (L1 Ranking)...")
    lasso = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10000)
    lasso.fit(X_train_sc, y_train_np)
    lasso_ranking = np.argsort(np.abs(lasso.coef_))[::-1]
    print(f"Lasso ranked features: {[feature_names[i] for i in lasso_ranking]}")
    lasso_subsets = {k: lasso_ranking[:k].tolist() for k in range(1, 14)}
    
    # -------------------------------------------------------------
    # 4. ElasticNet Ranking
    # -------------------------------------------------------------
    print("Running Method 4: ElasticNet Ranking...")
    enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.99], cv=5, 
                        random_state=RANDOM_STATE, max_iter=10000)
    enet.fit(X_train_sc, y_train_np)
    enet_ranking = np.argsort(np.abs(enet.coef_))[::-1]
    print(f"ElasticNet ranked features: {[feature_names[i] for i in enet_ranking]}")
    enet_subsets = {k: enet_ranking[:k].tolist() for k in range(1, 14)}
    
    # -------------------------------------------------------------
    # 2. Forward Sequential Feature Selection (Forward SFS)
    # -------------------------------------------------------------
    print("Running Method 2: Forward Sequential Feature Selection (Forward SFS)...")
    sfs_subsets = {}
    current_sfs = []
    remaining = list(range(num_features_total))
    
    for k in range(1, 14):
        best_cand = None
        best_cv_rmse = float('inf')
        for cand in remaining:
            trial = current_sfs + [cand]
            cv_rmse, _ = evaluate_features_cv(X_train_np, y_train_np, trial, splits)
            if cv_rmse < best_cv_rmse:
                best_cv_rmse = cv_rmse
                best_cand = cand
        current_sfs.append(best_cand)
        remaining.remove(best_cand)
        sfs_subsets[k] = list(current_sfs)
        
    # -------------------------------------------------------------
    # 3. Recursive Feature Elimination (RFE)
    # -------------------------------------------------------------
    print("Running Method 3: Recursive Feature Elimination (RFE)...")
    rfe_subsets = {}
    for k in range(1, 14):
        rfe = RFE(estimator=LinearRegression(), n_features_to_select=k)
        rfe.fit(X_train_sc, y_train_np)
        selected_idx = np.where(rfe.support_)[0].tolist()
        rfe_subsets[k] = selected_idx
        
    # -------------------------------------------------------------
    # 5. Exhaustive Feature Selection (Global Optimal Subsets)
    # -------------------------------------------------------------
    print("Running Method 5: Exhaustive Feature Selection across all 8,191 combinations...")
    exhaustive_subsets = {}
    for k in range(1, 14):
        combos = list(itertools.combinations(range(num_features_total), k))
        best_combo = None
        best_cv_rmse = float('inf')
        for combo in combos:
            combo_list = list(combo)
            cv_rmse, _ = evaluate_features_cv(X_train_np, y_train_np, combo_list, splits)
            if cv_rmse < best_cv_rmse:
                best_cv_rmse = cv_rmse
                best_combo = combo_list
        exhaustive_subsets[k] = best_combo
        print(f"k={k}: Best CV RMSE={best_cv_rmse:.4f} | {[feature_names[i] for i in best_combo]}")
        
    # -------------------------------------------------------------
    # Compile Results & Evaluate on Test Set
    # -------------------------------------------------------------
    methods_dict = {
        'Lasso (L1 Ranking)': lasso_subsets,
        'Forward SFS': sfs_subsets,
        'RFE': rfe_subsets,
        'ElasticNet Ranking': enet_subsets,
        'Exhaustive Search': exhaustive_subsets
    }
    
    results_records = []
    for method_name, subsets in methods_dict.items():
        for k in range(1, 14):
            idx_list = subsets[k]
            cv_rmse, cv_r2 = evaluate_features_cv(X_train_np, y_train_np, idx_list, splits)
            
            # Final test evaluation for this configuration
            scaler_k = StandardScaler()
            X_tr_sub_sc = scaler_k.fit_transform(X_train_np[:, idx_list])
            X_te_sub_sc = scaler_k.transform(X_test_np[:, idx_list])
            
            reg_k = LinearRegression()
            reg_k.fit(X_tr_sub_sc, y_train_np)
            test_preds = reg_k.predict(X_te_sub_sc)
            
            test_rmse = root_mean_squared_error(y_test_np, test_preds)
            test_r2 = r2_score(y_test_np, test_preds)
            
            feat_str = ", ".join([feature_names[i] for i in idx_list])
            results_records.append({
                'feature_selection_method': method_name,
                'num_features': k,
                'selected_features': feat_str,
                'cv_rmse': cv_rmse,
                'cv_r2': cv_r2,
                'test_rmse': test_rmse,
                'test_r2': test_r2
            })
            
    df_results = pd.DataFrame(results_records)
    return df_results, methods_dict


def plot_feature_selection_comparison(df_results: pd.DataFrame, output_dir: str):
    """
    Generate mandatory two-panel comparison figure:
    Left: RMSE Comparison (Lower is Better)
    Right: R-squared Comparison (Higher is Better)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    
    methods = df_results['feature_selection_method'].unique()
    palette = {
        'Lasso (L1 Ranking)': '#1f77b4',
        'Forward SFS': '#ff7f0e',
        'RFE': '#2ca02c',
        'ElasticNet Ranking': '#d62728',
        'Exhaustive Search': '#9467bd'
    }
    markers = {
        'Lasso (L1 Ranking)': 'o',
        'Forward SFS': 's',
        'RFE': '^',
        'ElasticNet Ranking': 'D',
        'Exhaustive Search': 'v'
    }
    
    # Left Chart: RMSE Comparison
    for method in methods:
        sub = df_results[df_results['feature_selection_method'] == method]
        ax1.plot(sub['num_features'], sub['cv_rmse'],
                 color=palette.get(method, '#333333'),
                 marker=markers.get(method, 'o'),
                 linewidth=2.2, markersize=7, label=method)
                 
    ax1.set_title("RMSE Comparison across 5 Feature Selection Methods", fontsize=14, weight='bold', pad=12)
    ax1.set_xlabel("Number of Features", fontsize=12, labelpad=8)
    ax1.set_ylabel("RMSE (Lower is Better)", fontsize=12, labelpad=8)
    ax1.set_xticks(range(1, 14))
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(title="Method", fontsize=10, title_fontsize=11, frameon=True, facecolor='white', framealpha=0.9)
    
    # Right Chart: R-squared Comparison
    for method in methods:
        sub = df_results[df_results['feature_selection_method'] == method]
        ax2.plot(sub['num_features'], sub['cv_r2'],
                 color=palette.get(method, '#333333'),
                 marker=markers.get(method, 'o'),
                 linewidth=2.2, markersize=7, label=method)
                 
    ax2.set_title("R-squared Comparison across 5 Feature Selection Methods", fontsize=14, weight='bold', pad=12)
    ax2.set_xlabel("Number of Features", fontsize=12, labelpad=8)
    ax2.set_ylabel("R-squared (Higher is Better)", fontsize=12, labelpad=8)
    ax2.set_xticks(range(1, 14))
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(title="Method", fontsize=10, title_fontsize=11, frameon=True, facecolor='white', framealpha=0.9)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "feature_selection_comparison.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved feature selection comparison figure: {fig_path}")


def train_and_evaluate_final_models(X_train: pd.DataFrame, y_train: pd.Series,
                                    X_test: pd.DataFrame, y_test: pd.Series,
                                    df_results: pd.DataFrame, output_dir: str, model_dir: str):
    """
    Train and evaluate final Multiple Linear Regression models:
    - Baseline: All 13 original predictors
    - Selected Optimal Model: Optimal configuration selected via cross-validation
    """
    print("\n" + "=" * 70)
    print("STEP 4: FINAL MODEL TRAINING, EVALUATION, AND VISUALIZATIONS")
    print("=" * 70)
    
    feature_names = X_train.columns.tolist()
    
    # Optimal configuration by CV RMSE
    best_row = df_results.loc[df_results['cv_rmse'].idxmin()]
    best_method = best_row['feature_selection_method']
    best_k = int(best_row['num_features'])
    best_feats = [f.strip() for f in best_row['selected_features'].split(',')]
    print(f"Optimal Model Selected via Cross-Validation:")
    print(f"Method: {best_method} | Features ({best_k}): {best_feats}")
    print(f"CV RMSE: {best_row['cv_rmse']:.4f} | CV R2: {best_row['cv_r2']:.4f}")
    
    # 1. Baseline Model (All 13 Features)
    scaler_all = StandardScaler()
    X_tr_all_sc = scaler_all.fit_transform(X_train)
    X_te_all_sc = scaler_all.transform(X_test)
    
    model_all = LinearRegression()
    model_all.fit(X_tr_all_sc, y_train)
    
    y_tr_pred_all = model_all.predict(X_tr_all_sc)
    y_te_pred_all = model_all.predict(X_te_all_sc)
    
    # 2. Selected Optimal Model
    scaler_sel = StandardScaler()
    X_tr_sel_sc = scaler_sel.fit_transform(X_train[best_feats])
    X_te_sel_sc = scaler_sel.transform(X_test[best_feats])
    
    model_sel = LinearRegression()
    model_sel.fit(X_tr_sel_sc, y_train)
    
    y_tr_pred_sel = model_sel.predict(X_tr_sel_sc)
    y_te_pred_sel = model_sel.predict(X_te_sel_sc)
    
    # Metrics
    metrics_records = [
        {
            'Model': 'Baseline (All 13 Predictors)',
            'Features_Count': 13,
            'Features': 'All 13 Predictors',
            'Train_R2': r2_score(y_train, y_tr_pred_all),
            'Train_MAE': mean_absolute_error(y_train, y_tr_pred_all),
            'Train_RMSE': root_mean_squared_error(y_train, y_tr_pred_all),
            'CV_RMSE': df_results[(df_results['feature_selection_method'] == 'Exhaustive Search') & 
                                  (df_results['num_features'] == 13)]['cv_rmse'].values[0],
            'CV_R2': df_results[(df_results['feature_selection_method'] == 'Exhaustive Search') & 
                                (df_results['num_features'] == 13)]['cv_r2'].values[0],
            'Test_R2': r2_score(y_test, y_te_pred_all),
            'Test_MAE': mean_absolute_error(y_test, y_te_pred_all),
            'Test_RMSE': root_mean_squared_error(y_test, y_te_pred_all)
        },
        {
            'Model': f'Selected Optimal ({best_method}, k={best_k})',
            'Features_Count': best_k,
            'Features': ", ".join(best_feats),
            'Train_R2': r2_score(y_train, y_tr_pred_sel),
            'Train_MAE': mean_absolute_error(y_train, y_tr_pred_sel),
            'Train_RMSE': root_mean_squared_error(y_train, y_tr_pred_sel),
            'CV_RMSE': best_row['cv_rmse'],
            'CV_R2': best_row['cv_r2'],
            'Test_R2': r2_score(y_test, y_te_pred_sel),
            'Test_MAE': mean_absolute_error(y_test, y_te_pred_sel),
            'Test_RMSE': root_mean_squared_error(y_test, y_te_pred_sel)
        }
    ]
    df_metrics = pd.DataFrame(metrics_records)
    metrics_path = os.path.join(output_dir, "model_metrics.csv")
    df_metrics.to_csv(metrics_path, index=False)
    print("\nModel Metrics Comparison:")
    print(df_metrics.to_string(index=False))
    
    # Save Model Artifacts
    os.makedirs(model_dir, exist_ok=True)
    artifacts = {
        'model_name': 'Multiple Linear Regression',
        'dataset': 'Boston Housing',
        'target_variable': 'MEDV',
        'selected_features': best_feats,
        'coefficients': model_sel.coef_.tolist(),
        'intercept': float(model_sel.intercept_),
        'scaler_mean': scaler_sel.mean_.tolist(),
        'scaler_scale': scaler_sel.scale_.tolist(),
        'all_features': feature_names,
        'baseline_coefficients': model_all.coef_.tolist(),
        'baseline_intercept': float(model_all.intercept_),
        'baseline_scaler_mean': scaler_all.mean_.tolist(),
        'baseline_scaler_scale': scaler_all.scale_.tolist(),
        'test_metrics': {
            'r2': float(r2_score(y_test, y_te_pred_sel)),
            'mae': float(mean_absolute_error(y_test, y_te_pred_sel)),
            'rmse': float(root_mean_squared_error(y_test, y_te_pred_sel))
        }
    }
    with open(os.path.join(model_dir, "model_artifacts.json"), "w") as f:
        json.dump(artifacts, f, indent=4)
        
    # Visualizations
    # 1. Actual vs Predicted Scatter Plot
    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(y_test, y_te_pred_sel, color='#2b5c8f', edgecolors='k', s=65, alpha=0.85, label='Test Data')
    min_val = min(y_test.min(), y_te_pred_sel.min()) * 0.95
    max_val = max(y_test.max(), y_te_pred_sel.max()) * 1.05
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y = x)')
    plt.title(f"Boston Housing: Actual vs. Predicted MEDV (Test Set)\n$R^2$ = {metrics_records[1]['Test_R2']:.4f} | RMSE = ${metrics_records[1]['Test_RMSE'] * 1000:,.0f}", 
              fontsize=13, weight='bold', pad=12)
    plt.xlabel("Actual MEDV ($1,000s)", fontsize=11)
    plt.ylabel("Predicted MEDV ($1,000s)", fontsize=11)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    act_pred_path = os.path.join(output_dir, "actual_vs_predicted.png")
    plt.savefig(act_pred_path, dpi=300)
    plt.close()
    
    # 2. Residual Distribution Plot
    residuals = y_test - y_te_pred_sel
    fig, (ax_res1, ax_res2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    sns.histplot(residuals, kde=True, ax=ax_res1, color='#d95f02', bins=12)
    ax_res1.axvline(0, color='black', linestyle='--', lw=1.5)
    ax_res1.set_title("Residual Distribution (Actual - Predicted)", fontsize=12, weight='bold')
    ax_res1.set_xlabel("Residual ($1,000s)", fontsize=11)
    ax_res1.set_ylabel("Count", fontsize=11)
    ax_res1.grid(True, linestyle='--', alpha=0.6)
    
    ax_res2.scatter(y_te_pred_sel, residuals, color='#d95f02', edgecolors='k', s=60, alpha=0.85)
    ax_res2.axhline(0, color='black', linestyle='--', lw=1.5)
    ax_res2.set_title("Residuals vs. Fitted Values (Homoscedasticity Check)", fontsize=12, weight='bold')
    ax_res2.set_xlabel("Fitted Values ($1,000s)", fontsize=11)
    ax_res2.set_ylabel("Residual ($1,000s)", fontsize=11)
    ax_res2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    res_path = os.path.join(output_dir, "residual_distribution.png")
    plt.savefig(res_path, dpi=300)
    plt.close()
    
    # 3. Model Coefficients Plot
    plt.figure(figsize=(9, 6), dpi=300)
    coef_series = pd.Series(model_sel.coef_, index=best_feats).sort_values()
    colors = ['#d95f02' if c < 0 else '#1b9e77' for c in coef_series.values]
    bars = plt.barh(coef_series.index, coef_series.values, color=colors, edgecolor='k', alpha=0.85)
    plt.axvline(0, color='black', lw=1)
    plt.title(f"Standardized Coefficients ({best_method}, k={best_k})", fontsize=13, weight='bold', pad=12)
    plt.xlabel("Standardized Weight", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    for bar in bars:
        w = bar.get_width()
        offset = 0.08 if w >= 0 else -0.35
        plt.text(w + offset, bar.get_y() + bar.get_height()/2, f"{w:+.2f}", 
                 va='center', fontsize=9, weight='bold')
                 
    plt.tight_layout()
    coef_path = os.path.join(output_dir, "model_coefficients.png")
    plt.savefig(coef_path, dpi=300)
    plt.close()
    print("Generated and saved all visualizations successfully.")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "boston.csv")
    output_dir = os.path.join(base_dir, "results")
    model_dir = os.path.join(base_dir, "models")
    
    # Step 1: Load and inspect
    df = load_and_inspect_data(data_path)
    
    # Step 2: EDA
    exploratory_data_analysis(df, output_dir)
    
    # Step 3: Train-test split (80/20)
    X = df.drop(columns=['MEDV'])
    y = df['MEDV']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"\nTraining set size: {len(X_train)} | Testing set size: {len(X_test)}")
    
    # Step 4: Run 5 Feature Selection methods
    df_results, _ = run_feature_selection(X_train, y_train, X_test, y_test)
    
    results_csv_path = os.path.join(output_dir, "feature_selection_comparison.csv")
    df_results.to_csv(results_csv_path, index=False)
    print(f"\nSaved feature selection comparison CSV: {results_csv_path}")
    
    # Step 5: Plot comparison figure
    plot_feature_selection_comparison(df_results, output_dir)
    
    # Step 6: Train final models & generate all remaining plots
    train_and_evaluate_final_models(X_train, y_train, X_test, y_test,
                                   df_results, output_dir, model_dir)
    print("\n" + "=" * 70)
    print("PROJECT 2 COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
