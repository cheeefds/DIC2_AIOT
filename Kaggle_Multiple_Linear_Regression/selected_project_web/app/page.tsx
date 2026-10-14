'use client';

import React, { useState, useMemo } from 'react';
import modelData from '@/model/model_data.json';

const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? '';
const imagePath = (fileName: string) => `${basePath}/images/${fileName}`;

// Feature metadata with ranges, defaults (mean), and human-friendly descriptions
interface FeatureMeta {
  name: string;
  label: string;
  category: string;
  min: number;
  max: number;
  step: number;
  unit: string;
  description: string;
}

const FEATURE_METADATA: Record<string, FeatureMeta> = {
  RM: {
    name: 'RM',
    label: '每戶平均房間數',
    category: '住宅特性',
    min: 3.5,
    max: 9.0,
    step: 0.1,
    unit: '間',
    description: '每戶住宅的平均房間數。',
  },
  LSTAT: {
    name: 'LSTAT',
    label: '低社經地位人口比例',
    category: '社經與教育',
    min: 1.5,
    max: 38.0,
    step: 0.5,
    unit: '%',
    description: '當地低社經地位人口的比例。',
  },
  PTRATIO: {
    name: 'PTRATIO',
    label: '學生教師比',
    category: '社經與教育',
    min: 12.0,
    max: 22.0,
    step: 0.2,
    unit: ':1',
    description: '城鎮學區中每位教師平均對應的學生人數。',
  },
  CRIM: {
    name: 'CRIM',
    label: '人均犯罪率',
    category: '社區與安全',
    min: 0.01,
    max: 50.0,
    step: 0.1,
    unit: '案件／人',
    description: '各城鎮的人均犯罪案件率。',
  },
  ZN: {
    name: 'ZN',
    label: '大型住宅用地比例（>25,000 平方英尺）',
    category: '社區與安全',
    min: 0.0,
    max: 100.0,
    step: 5.0,
    unit: '%',
    description: '劃為超過 25,000 平方英尺住宅用地的比例。',
  },
  B: {
    name: 'B',
    label: '人口統計指標 B',
    category: '社區與安全',
    min: 0.0,
    max: 396.9,
    step: 5.0,
    unit: '指數',
    description: '歷史人口統計指標 1000(Bk - 0.63)^2。',
  },
  DIS: {
    name: 'DIS',
    label: '至就業中心距離',
    category: '基礎建設與環境',
    min: 1.0,
    max: 12.0,
    step: 0.2,
    unit: '英里',
    description: '至五個主要都會就業中心的加權距離。',
  },
  RAD: {
    name: 'RAD',
    label: '放射狀公路可及性指數',
    category: '基礎建設與環境',
    min: 1,
    max: 24,
    step: 1,
    unit: '指數',
    description: '連接放射狀公路的可及性排名。',
  },
  TAX: {
    name: 'TAX',
    label: '每一萬美元房產稅率',
    category: '基礎建設與環境',
    min: 180,
    max: 720,
    step: 10,
    unit: '美元／萬美元',
    description: '每 10,000 美元完整估值所對應的房產稅率。',
  },
  NOX: {
    name: 'NOX',
    label: '氮氧化物濃度',
    category: '基礎建設與環境',
    min: 0.38,
    max: 0.88,
    step: 0.01,
    unit: 'ppm',
    description: '空氣中氮氧化物濃度。',
  },
};

// Preset scenarios for interactive exploration
const PRESETS: Record<string, { label: string; values: Record<string, number> }> = {
  average: {
    label: '波士頓郊區平均值',
    values: {
      RM: 6.3,
      LSTAT: 12.5,
      PTRATIO: 18.3,
      CRIM: 3.6,
      ZN: 11.5,
      B: 356.0,
      DIS: 3.8,
      RAD: 9,
      TAX: 404,
      NOX: 0.55,
    },
  },
  luxury: {
    label: '郊區高價住宅',
    values: {
      RM: 8.2,
      LSTAT: 3.5,
      PTRATIO: 14.5,
      CRIM: 0.05,
      ZN: 80.0,
      B: 395.0,
      DIS: 6.5,
      RAD: 4,
      TAX: 240,
      NOX: 0.42,
    },
  },
  urban: {
    label: '高密度市中心',
    values: {
      RM: 5.5,
      LSTAT: 24.0,
      PTRATIO: 20.2,
      CRIM: 12.5,
      ZN: 0.0,
      B: 320.0,
      DIS: 1.8,
      RAD: 24,
      TAX: 666,
      NOX: 0.72,
    },
  },
};

export default function Dashboard() {
  const selectedFeatures = modelData.selected_features;
  const coefficients = modelData.coefficients;
  const intercept = modelData.intercept;
  const scalerMean = modelData.scaler_mean;
  const scalerScale = modelData.scaler_scale;

  // State of feature input values initialized to mean
  const [featureValues, setFeatureValues] = useState<Record<string, number>>(() => {
    const initial: Record<string, number> = {};
    selectedFeatures.forEach((feat, idx) => {
      initial[feat] = parseFloat(scalerMean[idx].toFixed(2));
    });
    return initial;
  });

  const [activeTab, setActiveTab] = useState<'comparison' | 'actual_pred' | 'residuals' | 'heatmap' | 'coefficients'>('comparison');
  const [selectedPreset, setSelectedPreset] = useState<string>('average');

  // Compute live mathematical linear regression prediction
  const prediction = useMemo(() => {
    let predMedv = intercept;
    const contributions: Array<{
      feature: string;
      value: number;
      z: number;
      weight: number;
      contrib: number;
    }> = [];

    selectedFeatures.forEach((feat, idx) => {
      const val = featureValues[feat] !== undefined ? featureValues[feat] : scalerMean[idx];
      const z = (val - scalerMean[idx]) / scalerScale[idx];
      const contrib = coefficients[idx] * z;
      predMedv += contrib;

      contributions.push({
        feature: feat,
        value: val,
        z: z,
        weight: coefficients[idx],
        contrib: contrib,
      });
    });

    const dollarPrice = predMedv * 1000;
    const lowerDollar = (predMedv - 4.939) * 1000;
    const upperDollar = (predMedv + 4.939) * 1000;

    return {
      medv: predMedv,
      dollarPrice: Math.round(dollarPrice),
      lowerDollar: Math.round(lowerDollar),
      upperDollar: Math.round(upperDollar),
      isOutsideDatasetRange: predMedv < 5.0 || predMedv > 50.0,
      contributions: contributions.sort((a, b) => Math.abs(b.contrib) - Math.abs(a.contrib)),
    };
  }, [featureValues, selectedFeatures, coefficients, intercept, scalerMean, scalerScale]);

  const handleSliderChange = (feat: string, val: number) => {
    setSelectedPreset('custom');
    setFeatureValues((prev) => ({
      ...prev,
      [feat]: val,
    }));
  };

  const handlePresetSelect = (presetKey: string) => {
    setSelectedPreset(presetKey);
    const preset = PRESETS[presetKey];
    if (preset) {
      setFeatureValues(preset.values);
    }
  };

  // Group features by category
  const categories = useMemo(() => {
    const groups: Record<string, string[]> = {};
    selectedFeatures.forEach((feat) => {
      const meta = FEATURE_METADATA[feat];
      const cat = meta ? meta.category : '一般';
      if (!groups[cat]) groups[cat] = [];
      groups[cat].push(feat);
    });
    return groups;
  }, [selectedFeatures]);

  return (
    <main className="page-shell">
      {/* Header Banner */}
      <header style={{ marginBottom: '32px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', padding: '4px 12px', borderRadius: '9999px', background: 'rgba(59, 130, 246, 0.15)', border: '1px solid rgba(59, 130, 246, 0.3)', marginBottom: '12px' }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#3b82f6', display: 'inline-block' }}></span>
              <span style={{ fontSize: '0.8rem', fontWeight: 600, color: '#93c5fd', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Kaggle 迴歸分析專題 2 • Boston Housing
              </span>
            </div>
            <h1 style={{ fontSize: '2.5rem', fontWeight: 800, letterSpacing: '-0.02em', background: 'linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', marginBottom: '8px' }}>
              波士頓住宅價值預測
            </h1>
            <p style={{ color: 'var(--text-secondary)', fontSize: '1.05rem', maxWidth: '780px' }}>
              使用多元線性迴歸（Multiple Linear Regression）模型預測波士頓郊區住宅價格中位數；模型透過 8,191 組特徵子集合的完整交叉驗證，選出最佳的 10 個預測特徵。
            </p>
          </div>

          <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 500 }}>預設情境：</span>
            {Object.entries(PRESETS).map(([key, item]) => (
              <button
                key={key}
                onClick={() => handlePresetSelect(key)}
                className={selectedPreset === key ? 'btn-primary' : 'btn-secondary'}
                style={{ fontSize: '0.85rem', padding: '6px 14px' }}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* Model Performance Metric Cards */}
      <section style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px', marginBottom: '32px' }}>
        <div className="glass-panel" style={{ padding: '20px' }}>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600, marginBottom: '6px' }}>
            測試集判定係數（R²）
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 800, color: '#38bdf8' }} className="mono">
            0.6673
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
            可解釋 66.7% 的樣本外價格變異
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '20px' }}>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600, marginBottom: '6px' }}>
            測試集平均絕對誤差（MAE）
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 800, color: '#34d399' }} className="mono">
            $3,185
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
            平均絕對誤差為 3.185k 美元
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '20px' }}>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600, marginBottom: '6px' }}>
            五折交叉驗證均方根誤差（RMSE）
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 800, color: '#fbbf24' }} className="mono">
            4.8416
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
            8,191 組特徵子集合中的最低值
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '20px' }}>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600, marginBottom: '6px' }}>
            模型複雜度
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 800, color: '#a78bfa' }} className="mono">
            10 / 13
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
            精簡特徵集合（移除 INDUS、CHAS、AGE）
          </div>
        </div>
      </section>

      {/* Main Workspace: Left Controls vs. Right Live Prediction */}
      <div className="workspace-grid">
        
        {/* Left Column: Interactive Predictor Sliders */}
        <div className="glass-panel" style={{ padding: '28px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', borderBottom: '1px solid var(--border-color)', paddingBottom: '14px' }}>
            <div>
              <h2 style={{ fontSize: '1.25rem', fontWeight: 700 }}>互動式特徵控制</h2>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                調整住宅、社區與基礎建設參數，即時觀察模型估值變化。
              </p>
            </div>
            <button
              onClick={() => handlePresetSelect('average')}
              className="btn-secondary"
              style={{ fontSize: '0.8rem', padding: '6px 12px' }}
            >
              重設為平均值
            </button>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            {Object.entries(categories).map(([catTitle, feats]) => (
              <div key={catTitle} style={{ background: 'rgba(255, 255, 255, 0.02)', padding: '16px', borderRadius: '12px', border: '1px solid rgba(255, 255, 255, 0.04)' }}>
                <h3 style={{ fontSize: '0.9rem', fontWeight: 700, color: '#38bdf8', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: '14px' }}>
                  {catTitle}
                </h3>
                
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  {feats.map((feat) => {
                    const meta = FEATURE_METADATA[feat];
                    const val = featureValues[feat] !== undefined ? featureValues[feat] : 0;
                    return (
                      <div key={feat} className="slider-container">
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                          <label style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                            {meta ? meta.label : feat} ({feat})
                          </label>
                          <span className="mono" style={{ fontSize: '0.92rem', fontWeight: 700, color: '#06b6d4' }}>
                            {val.toLocaleString()} {meta?.unit}
                          </span>
                        </div>
                        <input
                          type="range"
                          min={meta ? meta.min : 0}
                          max={meta ? meta.max : 100}
                          step={meta ? meta.step : 1}
                          value={val}
                          aria-label={`${meta ? meta.label : feat}（${feat}）`}
                          onChange={(e) => handleSliderChange(feat, parseFloat(e.target.value))}
                        />
                        <div className="slider-scale">
                          <span>最小：{meta?.min}</span>
                          <span style={{ fontStyle: 'italic' }}>{meta?.description}</span>
                          <span>最大：{meta?.max}</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right Column: Live Real-Time Valuation Display & Feature Contribution */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          
          {/* Main Price Card */}
          <div className="glass-panel" style={{ padding: '28px', background: 'linear-gradient(145deg, rgba(17, 24, 39, 0.95) 0%, rgba(30, 41, 59, 0.85) 100%)', border: '1px solid rgba(59, 130, 246, 0.3)', position: 'relative', overflow: 'hidden' }}>
            <div style={{ position: 'absolute', top: '-40px', right: '-40px', width: '160px', height: '160px', background: 'radial-gradient(circle, rgba(6, 182, 212, 0.25) 0%, transparent 70%)', pointerEvents: 'none' }}></div>
            
            <div style={{ fontSize: '0.82rem', color: '#38bdf8', textTransform: 'uppercase', fontWeight: 700, letterSpacing: '0.06em', marginBottom: '8px' }}>
              即時模型估值
            </div>
            
            <div style={{ fontSize: '3rem', fontWeight: 800, color: '#ffffff', letterSpacing: '-0.03em', lineHeight: 1.1, marginBottom: '6px' }} className="mono">
              ${prediction.dollarPrice.toLocaleString()}
            </div>
            
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
              <span className="mono" style={{ fontSize: '1.1rem', color: '#94a3b8', fontWeight: 600 }}>
                MEDV: {prediction.medv.toFixed(2)}k
              </span>
              <span style={{ fontSize: '0.75rem', background: 'rgba(16, 185, 129, 0.15)', color: '#34d399', padding: '2px 8px', borderRadius: '4px', fontWeight: 600 }}>
                瀏覽器即時計算
              </span>
            </div>

            <div style={{ background: 'rgba(0, 0, 0, 0.3)', padding: '14px', borderRadius: '10px', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>
                <span>±1 RMSE 參考誤差範圍：</span>
                <span className="mono" style={{ color: '#f1f5f9', fontWeight: 600 }}>
                  ${prediction.lowerDollar.toLocaleString()} – ${prediction.upperDollar.toLocaleString()}
                </span>
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                此範圍為預測值加減一次測試集 RMSE（4,939 美元），不是 95% 信賴區間。
              </div>
            </div>
            {prediction.isOutsideDatasetRange && (
              <p className="extrapolation-warning">
                此結果超出資料集 MEDV 的 5k–50k 美元範圍，屬於外插預測，解讀時請特別留意。
              </p>
            )}
          </div>

          {/* Feature Contribution Breakdown */}
          <div className="glass-panel" style={{ padding: '24px', flex: 1 }}>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '8px' }}>
              特徵影響拆解（相對基準平均值的 Δ）
            </h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
              基準截距 = ${Math.round(intercept * 1000).toLocaleString()}（MEDV {intercept.toFixed(2)}k）。長條顯示各標準化特徵對預測值的推動幅度（$w_j \times z_j$）。
            </p>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }} className="custom-scrollbar">
              {prediction.contributions.map((item) => {
                const isPos = item.contrib >= 0;
                const absDollar = Math.round(Math.abs(item.contrib) * 1000);
                const percentBar = Math.min(100, (Math.abs(item.contrib) / 8.0) * 100);

                return (
                  <div key={item.feature} style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem' }}>
                      <span style={{ fontWeight: 600, color: '#f8fafc' }}>
                        {item.feature}
                        <span style={{ color: 'var(--text-muted)', fontWeight: 400, marginLeft: '6px' }}>
                          (z = {item.z > 0 ? `+${item.z.toFixed(2)}` : item.z.toFixed(2)})
                        </span>
                      </span>
                      <span className="mono" style={{ fontWeight: 700, color: isPos ? '#34d399' : '#f87171' }}>
                        {isPos ? `+$${absDollar.toLocaleString()}` : `-$${absDollar.toLocaleString()}`}
                      </span>
                    </div>

                    <div style={{ width: '100%', height: '6px', background: 'rgba(255, 255, 255, 0.06)', borderRadius: '3px', overflow: 'hidden' }}>
                      <div
                        style={{
                          width: `${percentBar}%`,
                          height: '100%',
                          background: isPos
                            ? 'linear-gradient(90deg, #10b981, #06b6d4)'
                            : 'linear-gradient(90deg, #f43f5e, #fb923c)',
                          borderRadius: '3px',
                          transition: 'width 0.2s ease',
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Model Formulation & Scientific Explanation */}
      <section className="glass-panel" style={{ padding: '28px', marginBottom: '40px' }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '12px' }}>
          數學公式與推論流程
        </h2>
        <div className="formula-grid">
          <div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.92rem', marginBottom: '12px' }}>
              模型僅使用訓練集（N=404，seed 42）估計權重與標準化參數，再進行預測，以確保<strong>沒有資料洩漏（data leakage）</strong>：
            </p>
            <div style={{ background: 'rgba(0, 0, 0, 0.35)', padding: '14px', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.05)' }} className="mono">
              <span style={{ color: '#38bdf8' }}>MEDV_hat</span> = β₀ + Σ [ βⱼ · ((xⱼ - μⱼ) / σⱼ) ]<br />
              <span style={{ color: '#94a3b8', fontSize: '0.8rem' }}>
                其中 β₀ = 22.80；μⱼ 與 σⱼ 分別為訓練集特徵的平均值與標準差。
              </span>
            </div>
          </div>

          <div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.92rem', marginBottom: '12px' }}>
              <strong>特徵選擇依據：</strong>針對 13 個原始特徵的 8,191 種可能組合進行五折交叉驗證後，移除 <code>INDUS</code>、<code>CHAS</code> 與 <code>AGE</code> 可得到最低的驗證 RMSE：
            </p>
            <ul style={{ color: 'var(--text-muted)', fontSize: '0.85rem', paddingLeft: '18px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <li><code>INDUS</code>（工業用地比例）與 <code>TAX</code>、<code>NOX</code> 存在較高共線性（r &gt; 0.70）。</li>
              <li>加入距離特徵 <code>DIS</code> 後，<code>AGE</code>（老屋比例）提供的資訊較為重複。</li>
              <li><code>CHAS</code>（是否鄰近 Charles River）在 93% 的資料中數值相同，變異很低。</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Experimental Results & Visualizations Gallery */}
      <section className="glass-panel" style={{ padding: '28px', marginBottom: '40px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '14px', marginBottom: '20px' }}>
          <div>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 700 }}>
              實驗圖表與模型比較
            </h2>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              以下為實驗訓練流程直接產生的 300 DPI 高解析度分析圖表。
            </p>
          </div>

          {/* Tab Selector */}
          <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', background: 'rgba(0,0,0,0.3)', padding: '4px', borderRadius: '10px' }}>
            <button
              onClick={() => setActiveTab('comparison')}
              className={activeTab === 'comparison' ? 'btn-primary' : 'btn-secondary'}
              style={{ fontSize: '0.8rem', padding: '6px 12px' }}
            >
              五種特徵選擇方法
            </button>
            <button
              onClick={() => setActiveTab('actual_pred')}
              className={activeTab === 'actual_pred' ? 'btn-primary' : 'btn-secondary'}
              style={{ fontSize: '0.8rem', padding: '6px 12px' }}
            >
              實際值與預測值
            </button>
            <button
              onClick={() => setActiveTab('residuals')}
              className={activeTab === 'residuals' ? 'btn-primary' : 'btn-secondary'}
              style={{ fontSize: '0.8rem', padding: '6px 12px' }}
            >
              殘差診斷
            </button>
            <button
              onClick={() => setActiveTab('coefficients')}
              className={activeTab === 'coefficients' ? 'btn-primary' : 'btn-secondary'}
              style={{ fontSize: '0.8rem', padding: '6px 12px' }}
            >
              模型係數
            </button>
            <button
              onClick={() => setActiveTab('heatmap')}
              className={activeTab === 'heatmap' ? 'btn-primary' : 'btn-secondary'}
              style={{ fontSize: '0.8rem', padding: '6px 12px' }}
            >
              相關係數熱圖
            </button>
          </div>
        </div>

        {/* Tab Content */}
        <div style={{ background: '#070a12', borderRadius: '12px', padding: '16px', border: '1px solid rgba(255, 255, 255, 0.05)', textAlign: 'center' }}>
          {activeTab === 'comparison' && (
            <div>
              <img
                src={imagePath('feature_selection_comparison.png')}
                alt="五種特徵選擇方法比較圖"
                style={{ width: '100%', maxHeight: '600px', objectFit: 'contain', borderRadius: '8px' }}
              />
              <p style={{ marginTop: '12px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                <strong>圖 1：</strong>比較五種特徵選擇策略：最小絕對壓縮與選擇運算子（Lasso）、前向循序特徵選擇（Forward SFS）、遞迴特徵消除（RFE）、彈性網路（ElasticNet）與完整搜尋（Exhaustive Search），並呈現 1 至 13 個特徵下的五折交叉驗證 RMSE（左）與 R²（右）。
              </p>
            </div>
          )}

          {activeTab === 'actual_pred' && (
            <div>
              <img
                src={imagePath('actual_vs_predicted.png')}
                alt="實際值與預測值比較圖"
                style={{ width: '100%', maxHeight: '560px', objectFit: 'contain', borderRadius: '8px' }}
              />
              <p style={{ marginTop: '12px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                <strong>圖 2：</strong>測試集預測值與實際住宅價格中位數的比較；45 度線代表完全正確的預測（R² = 0.6673）。
              </p>
            </div>
          )}

          {activeTab === 'residuals' && (
            <div>
              <img
                src={imagePath('residual_distribution.png')}
                alt="殘差分布與診斷圖"
                style={{ width: '100%', maxHeight: '560px', objectFit: 'contain', borderRadius: '8px' }}
              />
              <p style={{ marginTop: '12px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                <strong>圖 3：</strong>殘差直方圖（左）與殘差對擬合值的散布診斷圖（右），用於觀察誤差分布與變異情形。
              </p>
            </div>
          )}

          {activeTab === 'coefficients' && (
            <div>
              <img
                src={imagePath('model_coefficients.png')}
                alt="模型標準化係數圖"
                style={{ width: '100%', maxHeight: '560px', objectFit: 'contain', borderRadius: '8px' }}
              />
              <p style={{ marginTop: '12px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                <strong>圖 4：</strong>10 個入選特徵的標準化係數，顯示各特徵影響方向與相對權重。
              </p>
            </div>
          )}

          {activeTab === 'heatmap' && (
            <div>
              <img
                src={imagePath('correlation_heatmap.png')}
                alt="相關係數熱圖"
                style={{ width: '100%', maxHeight: '600px', objectFit: 'contain', borderRadius: '8px' }}
              />
              <p style={{ marginTop: '12px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                <strong>圖 5：</strong>13 個特徵與目標變數 MEDV 的下三角 Pearson 相關係數矩陣。
              </p>
            </div>
          )}
        </div>
      </section>

      {/* Footer */}
      <footer style={{ borderTop: '1px solid var(--border-color)', paddingTop: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
        <div>
          Kaggle 多元線性迴歸專題 • 成果展示網站
        </div>
        <div>
          Next.js 14 • TypeScript • 無資料洩漏的機器學習流程（ML Pipeline）• GitHub Pages
        </div>
      </footer>
    </main>
  );
}
