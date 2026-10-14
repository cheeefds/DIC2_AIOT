# 波士頓住宅價值預測網站

這個 Next.js 14 網站展示 Boston Housing 多元線性迴歸實驗，包含即時預測、特徵影響拆解、模型評估指標與實驗圖表。

## 功能

- 以 10 個入選特徵在瀏覽器中即時計算 MEDV 預測值。
- 提供平均郊區、高價住宅與高密度市中心三種預設情境。
- 顯示各特徵對預測結果的正向或負向影響。
- 整合特徵選擇、實際值與預測值、殘差、模型係數及相關係數圖表。
- 支援桌面、平板與手機畫面。

## 本機執行

需求：Node.js 20 以上版本與 npm。

```powershell
cd Kaggle_Multiple_Linear_Regression\selected_project_web
npm ci
npm run dev
```

開啟 <http://localhost:3000>。

## 建置與驗證

一般本機靜態建置：

```powershell
npm run build
npm run verify:config
npm run verify:export
```

模擬 GitHub Pages repository 子路徑建置：

```powershell
$env:GITHUB_ACTIONS='true'
$env:GITHUB_REPOSITORY='your-account/your-repository'
npm run build
npm run verify:export
```

輸出檔案位於 `out/`。GitHub Actions 會自動設定實際 repository 名稱，因此不需要手動修改 `basePath`。

## 為什麼可以部署到 GitHub Pages

GitHub Pages 只能提供 HTML、CSS、JavaScript 與圖片等靜態檔案，不能執行 Next.js Server API。本網站的迴歸計算已在瀏覽器端完成，因此採用 `output: 'export'`，不需要後端伺服器。

## 主要檔案

```text
selected_project_web/
├── app/
│   ├── globals.css       # 全站樣式與響應式版面
│   ├── layout.tsx        # SEO metadata 與語言設定
│   └── page.tsx          # 儀表板與即時預測
├── model/model_data.json # 模型係數與標準化參數
├── public/images/        # 實驗圖表
├── scripts/              # 靜態輸出驗證
└── next.config.mjs       # GitHub Pages 靜態匯出設定
```

> 本網站為課程與研究展示用途。預測結果反映此資料集與模型設定，不應作為真實房地產估價依據。
