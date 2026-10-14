"""
Project 1: 50 Startups Profit Prediction using Multiple Linear Regression
Manual Machine Learning Pipeline Implementation
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
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

# Set random seed for full reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Configure Matplotlib styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def load_and_clean_data(csv_path: str):
    """Load dataset, inspect statistics, and check for missing values."""
    print("=" * 70)
    print("STEP 1: DATASET LOADING AND INSPECTION")
    print("=" * 70)
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nSummary Statistics:")
    print(df.describe())
    return df


def exploratory_data_analysis(df: pd.DataFrame, output_dir: str):
    """Perform exploratory data analysis and generate correlation heatmap."""
    print("\n" + "=" * 70)
    print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)
    os.makedirs(output_dir, exist_ok=True)
    
    # Correlation heatmap of numeric variables
    num_cols = ['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']
    corr = df[num_cols].corr()
    print("Correlation with Profit:")
    print(corr['Profit'].sort_values(ascending=False))
    
    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="Blues", cbar=True, square=True,
                linewidths=1, linecolor="white", annot_kws={"size": 11, "weight": "bold"})
    plt.title("50 Startups: Correlation Heatmap", fontsize=14, pad=15, weight='bold')
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Saved correlation heatmap: {heatmap_path}")


def prepare_features(df: pd.DataFrame):
    """
    Encode categorical features and define original predictor groups.
    Original predictors:
    - 'R&D Spend'
    - 'Administration'
    - 'Marketing Spend'
    - 'State' (one-hot encoded, avoiding dummy variable trap with drop_first=True)
    """
    df_encoded = pd.get_dummies(df, columns=['State'], drop_first=True, dtype=float)
    state_dummy_cols = [c for c in df_encoded.columns if c.startswith('State_')]
    
    feature_groups = {
        'R&D Spend': ['R&D Spend'],
        'Administration': ['Administration'],
        'Marketing Spend': ['Marketing Spend'],
        'State': state_dummy_cols
    }
    
    original_features = ['R&D Spend', 'Administration', 'Marketing Spend', 'State']
    all_encoded_cols = ['R&D Spend', 'Administration', 'Marketing Spend'] + state_dummy_cols
    
    X = df_encoded[all_encoded_cols]
    y = df_encoded['Profit']
    return X, y, original_features, feature_groups


def evaluate_feature_subset_cv(X_tr: pd.DataFrame, y_tr: pd.Series, 
                               cols: list, kf: KFold):
    """Evaluate a linear regression model with a specific column subset using CV."""
    rmse_scores = []
    r2_scores = []
    
    for train_idx, val_idx in kf.split(X_tr):
        X_fold_tr, y_fold_tr = X_tr.iloc[train_idx][cols], y_tr.iloc[train_idx]
        X_fold_va, y_fold_va = X_tr.iloc[val_idx][cols], y_tr.iloc[val_idx]
        
        scaler = StandardScaler()
        X_fold_tr_sc = scaler.fit_transform(X_fold_tr)
        X_fold_va_sc = scaler.transform(X_fold_va)
        
        model = LinearRegression()
        model.fit(X_fold_tr_sc, y_fold_tr)
        preds = model.predict(X_fold_va_sc)
        
        fold_rmse = root_mean_squared_error(y_fold_va, preds)
        fold_r2 = r2_score(y_fold_va, preds)
        rmse_scores.append(fold_rmse)
        r2_scores.append(fold_r2)
        
    return np.mean(rmse_scores), np.mean(r2_scores)


def run_feature_selection(X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series,
                          original_features: list, feature_groups: dict):
    """
    Execute 5 feature selection methods over 1 to 4 original predictors:
    1. Lasso (L1 Ranking)
    2. Forward Sequential Feature Selection (Forward SFS)
    3. Recursive Feature Elimination (RFE)
    4. ElasticNet Ranking
    5. Exhaustive Feature Selection (Exhaustive Search)
    """
    print("\n" + "=" * 70)
    print("STEP 3: FEATURE SELECTION COMPARISON ACROSS 5 METHODS")
    print("=" * 70)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Pre-scale entire training set for ranking methods
    scaler_full = StandardScaler()
    X_train_sc = pd.DataFrame(scaler_full.fit_transform(X_train), 
                              columns=X_train.columns, index=X_train.index)
    
    # -------------------------------------------------------------
    # Method 1: Lasso (L1 Ranking)
    # -------------------------------------------------------------
    lasso = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10000)
    lasso.fit(X_train_sc, y_train)
    lasso_coef_series = pd.Series(lasso.coef_, index=X_train.columns)
    
    # Group importance = L2 norm of standardized coefficients
    lasso_orig_imp = {}
    for orig in original_features:
        cols = feature_groups[orig]
        lasso_orig_imp[orig] = np.linalg.norm(lasso_coef_series[cols].values)
    lasso_ranked = sorted(lasso_orig_imp.keys(), key=lambda k: lasso_orig_imp[k], reverse=True)
    print(f"Lasso Ranked Predictors: {lasso_ranked}")
    
    # -------------------------------------------------------------
    # Method 4: ElasticNet Ranking
    # -------------------------------------------------------------
    enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.99], cv=5, 
                        random_state=RANDOM_STATE, max_iter=10000)
    enet.fit(X_train_sc, y_train)
    enet_coef_series = pd.Series(enet.coef_, index=X_train.columns)
    
    enet_orig_imp = {}
    for orig in original_features:
        cols = feature_groups[orig]
        enet_orig_imp[orig] = np.linalg.norm(enet_coef_series[cols].values)
    enet_ranked = sorted(enet_orig_imp.keys(), key=lambda k: enet_orig_imp[k], reverse=True)
    print(f"ElasticNet Ranked Predictors: {enet_ranked}")
    
    # -------------------------------------------------------------
    # Method 2: Forward Sequential Feature Selection (Forward SFS)
    # -------------------------------------------------------------
    sfs_selected_by_k = {}
    current_selected = []
    remaining = list(original_features)
    
    for k in range(1, 5):
        best_candidate = None
        best_cv_rmse = float('inf')
        for cand in remaining:
            trial = current_selected + [cand]
            trial_cols = [c for feat in trial for c in feature_groups[feat]]
            cv_rmse, _ = evaluate_feature_subset_cv(X_train, y_train, trial_cols, kf)
            if cv_rmse < best_cv_rmse:
                best_cv_rmse = cv_rmse
                best_candidate = cand
        current_selected.append(best_candidate)
        remaining.remove(best_candidate)
        sfs_selected_by_k[k] = list(current_selected)
    print(f"Forward SFS by k: {sfs_selected_by_k}")
    
    # -------------------------------------------------------------
    # Method 3: Recursive Feature Elimination (RFE)
    # -------------------------------------------------------------
    rfe_selected_by_k = {}
    current_rfe = list(original_features)
    rfe_selected_by_k[4] = list(current_rfe)
    
    for k in range(3, 0, -1):
        # Fit OLS on current set to determine weakest original feature
        curr_cols = [c for feat in current_rfe for c in feature_groups[feat]]
        scaler = StandardScaler()
        X_sub_sc = scaler.fit_transform(X_train[curr_cols])
        ols = LinearRegression()
        ols.fit(X_sub_sc, y_train)
        ols_coef_series = pd.Series(ols.coef_, index=curr_cols)
        
        # Calculate importance for each feature currently in set
        feat_imp = {}
        for feat in current_rfe:
            cols = feature_groups[feat]
            feat_imp[feat] = np.linalg.norm(ols_coef_series[cols].values)
            
        weakest = min(feat_imp.keys(), key=lambda k: feat_imp[k])
        current_rfe.remove(weakest)
        rfe_selected_by_k[k] = list(current_rfe)
    print(f"RFE by k: {rfe_selected_by_k}")
    
    # -------------------------------------------------------------
    # Method 5: Exhaustive Feature Selection
    # -------------------------------------------------------------
    exhaustive_selected_by_k = {}
    for k in range(1, 5):
        combos = list(itertools.combinations(original_features, k))
        best_combo = None
        best_cv_rmse = float('inf')
        for combo in combos:
            trial_cols = [c for feat in combo for c in feature_groups[feat]]
            cv_rmse, _ = evaluate_feature_subset_cv(X_train, y_train, trial_cols, kf)
            if cv_rmse < best_cv_rmse:
                best_cv_rmse = cv_rmse
                best_combo = combo
        exhaustive_selected_by_k[k] = list(best_combo)
    print(f"Exhaustive Search by k: {exhaustive_selected_by_k}")
    
    # -------------------------------------------------------------
    # Evaluate All Methods across k=1..4 with 5-Fold CV
    # -------------------------------------------------------------
    results_records = []
    
    methods = {
        'Lasso (L1 Ranking)': [lasso_ranked[:k] for k in range(1, 5)],
        'Forward SFS': [sfs_selected_by_k[k] for k in range(1, 5)],
        'RFE': [rfe_selected_by_k[k] for k in range(1, 5)],
        'ElasticNet Ranking': [enet_ranked[:k] for k in range(1, 5)],
        'Exhaustive Search': [exhaustive_selected_by_k[k] for k in range(1, 5)]
    }
    
    for method_name, selected_list in methods.items():
        for k_idx, feat_subset in enumerate(selected_list):
            k = k_idx + 1
            cols = [c for feat in feat_subset for c in feature_groups[feat]]
            cv_rmse, cv_r2 = evaluate_feature_subset_cv(X_train, y_train, cols, kf)
            
            # Final test evaluation for this configuration
            scaler_final = StandardScaler()
            X_tr_sc = scaler_final.fit_transform(X_train[cols])
            X_te_sc = scaler_final.transform(X_test[cols])
            model_final = LinearRegression()
            model_final.fit(X_tr_sc, y_train)
            test_preds = model_final.predict(X_te_sc)
            test_rmse = root_mean_squared_error(y_test, test_preds)
            test_r2 = r2_score(y_test, test_preds)
            
            results_records.append({
                'feature_selection_method': method_name,
                'num_features': k,
                'selected_features': ", ".join(feat_subset),
                'cv_rmse': cv_rmse,
                'cv_r2': cv_r2,
                'test_rmse': test_rmse,
                'test_r2': test_r2
            })
            
    df_results = pd.DataFrame(results_records)
    return df_results


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
                 linewidth=2.2, markersize=8, label=method)
                 
    ax1.set_title("RMSE Comparison across 5 Feature Selection Methods", fontsize=14, weight='bold', pad=12)
    ax1.set_xlabel("Number of Features", fontsize=12, labelpad=8)
    ax1.set_ylabel("RMSE (Lower is Better)", fontsize=12, labelpad=8)
    ax1.set_xticks(sorted(df_results['num_features'].unique()))
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(title="Method", fontsize=10, title_fontsize=11, frameon=True, facecolor='white', framealpha=0.9)
    
    # Right Chart: R-squared Comparison
    for method in methods:
        sub = df_results[df_results['feature_selection_method'] == method]
        ax2.plot(sub['num_features'], sub['cv_r2'],
                 color=palette.get(method, '#333333'),
                 marker=markers.get(method, 'o'),
                 linewidth=2.2, markersize=8, label=method)
                 
    ax2.set_title("R-squared Comparison across 5 Feature Selection Methods", fontsize=14, weight='bold', pad=12)
    ax2.set_xlabel("Number of Features", fontsize=12, labelpad=8)
    ax2.set_ylabel("R-squared (Higher is Better)", fontsize=12, labelpad=8)
    ax2.set_xticks(sorted(df_results['num_features'].unique()))
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(title="Method", fontsize=10, title_fontsize=11, frameon=True, facecolor='white', framealpha=0.9)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "feature_selection_comparison.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved feature selection comparison figure: {fig_path}")


def train_and_evaluate_final_models(X_train: pd.DataFrame, y_train: pd.Series,
                                    X_test: pd.DataFrame, y_test: pd.Series,
                                    feature_groups: dict, df_results: pd.DataFrame,
                                    output_dir: str, model_dir: str):
    """
    Train and evaluate final Multiple Linear Regression model:
    - Baseline model: All original features
    - Selected optimal model based on lowest cross-validation RMSE
    """
    print("\n" + "=" * 70)
    print("STEP 4: FINAL MODEL TRAINING, EVALUATION, AND VISUALIZATIONS")
    print("=" * 70)
    
    # Find best configuration based on CV RMSE
    best_row = df_results.loc[df_results['cv_rmse'].idxmin()]
    best_method = best_row['feature_selection_method']
    best_k = int(best_row['num_features'])
    best_feats = [f.strip() for f in best_row['selected_features'].split(',')]
    print(f"Optimal Model Selected via CV:")
    print(f"Method: {best_method} | Features ({best_k}): {best_feats}")
    print(f"CV RMSE: {best_row['cv_rmse']:.2f} | CV R2: {best_row['cv_r2']:.4f}")
    
    # 1. Baseline Model (All Features)
    all_cols = X_train.columns.tolist()
    scaler_all = StandardScaler()
    X_tr_all_sc = scaler_all.fit_transform(X_train[all_cols])
    X_te_all_sc = scaler_all.transform(X_test[all_cols])
    
    model_all = LinearRegression()
    model_all.fit(X_tr_all_sc, y_train)
    
    y_tr_pred_all = model_all.predict(X_tr_all_sc)
    y_te_pred_all = model_all.predict(X_te_all_sc)
    
    # 2. Selected Optimal Model
    sel_cols = [c for feat in best_feats for c in feature_groups[feat]]
    scaler_sel = StandardScaler()
    X_tr_sel_sc = scaler_sel.fit_transform(X_train[sel_cols])
    X_te_sel_sc = scaler_sel.transform(X_test[sel_cols])
    
    model_sel = LinearRegression()
    model_sel.fit(X_tr_sel_sc, y_train)
    
    y_tr_pred_sel = model_sel.predict(X_tr_sel_sc)
    y_te_pred_sel = model_sel.predict(X_te_sel_sc)
    
    # Compute Metrics
    metrics_records = [
        {
            'Model': 'Baseline (All 4 Features)',
            'Features_Count': len(feature_groups),
            'Features': 'All Features',
            'Train_R2': r2_score(y_train, y_tr_pred_all),
            'Train_MAE': mean_absolute_error(y_train, y_tr_pred_all),
            'Train_RMSE': root_mean_squared_error(y_train, y_tr_pred_all),
            'CV_RMSE': df_results[(df_results['feature_selection_method'] == 'Exhaustive Search') & 
                                  (df_results['num_features'] == 4)]['cv_rmse'].values[0],
            'CV_R2': df_results[(df_results['feature_selection_method'] == 'Exhaustive Search') & 
                                (df_results['num_features'] == 4)]['cv_r2'].values[0],
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
        'dataset': '50 Startups',
        'selected_features': best_feats,
        'encoded_columns': sel_cols,
        'coefficients': model_sel.coef_.tolist(),
        'intercept': float(model_sel.intercept_),
        'scaler_mean': scaler_sel.mean_.tolist(),
        'scaler_scale': scaler_sel.scale_.tolist(),
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
    plt.scatter(y_test, y_te_pred_sel, color='#1f77b4', edgecolors='k', s=80, alpha=0.85, label='Test Observations')
    min_val = min(y_test.min(), y_te_pred_sel.min()) * 0.95
    max_val = max(y_test.max(), y_te_pred_sel.max()) * 1.05
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y = x)')
    plt.title(f"Actual vs. Predicted Profit (Test Set)\n$R^2$ = {metrics_records[1]['Test_R2']:.4f} | RMSE = ${metrics_records[1]['Test_RMSE']:,.2f}", 
              fontsize=13, weight='bold', pad=12)
    plt.xlabel("Actual Profit ($)", fontsize=11)
    plt.ylabel("Predicted Profit ($)", fontsize=11)
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
    
    # Residual Histogram + KDE
    sns.histplot(residuals, kde=True, ax=ax_res1, color='#2ca02c', bins=8)
    ax_res1.axvline(0, color='r', linestyle='--', lw=1.5)
    ax_res1.set_title("Residual Distribution (Actual - Predicted)", fontsize=12, weight='bold')
    ax_res1.set_xlabel("Residual ($)", fontsize=11)
    ax_res1.set_ylabel("Count", fontsize=11)
    ax_res1.grid(True, linestyle='--', alpha=0.6)
    
    # Residual vs Predicted
    ax_res2.scatter(y_te_pred_sel, residuals, color='#2ca02c', edgecolors='k', s=70, alpha=0.85)
    ax_res2.axhline(0, color='r', linestyle='--', lw=1.5)
    ax_res2.set_title("Residuals vs. Fitted Values", fontsize=12, weight='bold')
    ax_res2.set_xlabel("Fitted Values ($)", fontsize=11)
    ax_res2.set_ylabel("Residual ($)", fontsize=11)
    ax_res2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    res_path = os.path.join(output_dir, "residual_distribution.png")
    plt.savefig(res_path, dpi=300)
    plt.close()
    
    # 3. Model Coefficients Plot
    plt.figure(figsize=(8, 5), dpi=300)
    coef_series = pd.Series(model_sel.coef_, index=sel_cols)
    colors = ['#1f77b4' if c > 0 else '#d62728' for c in coef_series.values]
    bars = plt.barh(coef_series.index, coef_series.values, color=colors, edgecolor='k', alpha=0.85)
    plt.axvline(0, color='black', lw=1)
    plt.title("Standardized Coefficients of Selected Model", fontsize=13, weight='bold', pad=12)
    plt.xlabel("Standardized Coefficient Weight", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add numerical labels
    for bar in bars:
        w = bar.get_width()
        offset = 500 if w >= 0 else -1500
        plt.text(w + offset, bar.get_y() + bar.get_height()/2, f"{w:,.1f}", 
                 va='center', fontsize=9, weight='bold')
                 
    plt.tight_layout()
    coef_path = os.path.join(output_dir, "model_coefficients.png")
    plt.savefig(coef_path, dpi=300)
    plt.close()
    print("Generated and saved all visualizations successfully.")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "50_Startups.csv")
    output_dir = os.path.join(base_dir, "results")
    model_dir = os.path.join(base_dir, "models")
    
    # Step 1: Load and inspect
    df = load_and_clean_data(data_path)
    
    # Step 2: EDA
    exploratory_data_analysis(df, output_dir)
    
    # Step 3: Prepare features & split (80/20 train/test)
    X, y, original_features, feature_groups = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")
    
    # Step 4: Run 5 Feature Selection methods
    df_results = run_feature_selection(X_train, y_train, X_test, y_test,
                                       original_features, feature_groups)
    
    results_csv_path = os.path.join(output_dir, "feature_selection_comparison.csv")
    df_results.to_csv(results_csv_path, index=False)
    print(f"\nSaved feature selection comparison CSV: {results_csv_path}")
    
    # Step 5: Plot comparison figure
    plot_feature_selection_comparison(df_results, output_dir)
    
    # Step 6: Train final models & generate all remaining plots
    train_and_evaluate_final_models(X_train, y_train, X_test, y_test,
                                   feature_groups, df_results,
                                   output_dir, model_dir)
    print("\n" + "=" * 70)
    print("PROJECT 1 COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
