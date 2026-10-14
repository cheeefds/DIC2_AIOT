import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const outputDirectory = resolve('out');
const indexPath = resolve(outputDirectory, 'index.html');
const [repositoryOwner = '', repositoryName = ''] = (process.env.GITHUB_REPOSITORY ?? '').split('/');
const isGitHubPagesBuild = process.env.GITHUB_ACTIONS === 'true' && repositoryName.length > 0;
const isUserOrOrganizationSite = repositoryName.toLowerCase() === `${repositoryOwner}.github.io`.toLowerCase();
const expectedBasePath = isGitHubPagesBuild && !isUserOrOrganizationSite ? `/${repositoryName}` : '';

assert.ok(existsSync(indexPath), '找不到 out/index.html，網站尚未完成靜態匯出。');

const html = readFileSync(indexPath, 'utf8');

assert.match(html, /<html lang="zh-Hant">/, '頁面語言必須設定為繁體中文。');
assert.match(html, /波士頓住宅價值預測/, '靜態首頁缺少繁體中文主標題。');
assert.ok(html.includes('Project 1：50 Startups'), '問題選單缺少 Project 1。');
assert.ok(html.includes('Project 2：Boston Housing'), '問題選單缺少 Project 2。');
assert.ok(!html.includes('95% 估計區間'), '不得將 ±1 RMSE 誤稱為 95% 估計區間。');
assert.ok(html.includes('±1 RMSE 參考誤差範圍'), '頁面必須正確標示 ±1 RMSE 參考誤差範圍。');
if (expectedBasePath) {
  assert.ok(
    html.includes(`${expectedBasePath}/_next/`),
    'Next.js 靜態資源未使用 GitHub Pages repository 子路徑。',
  );
  assert.ok(
    html.includes(`${expectedBasePath}/images/feature_selection_comparison.png`),
    '分析圖片未使用 GitHub Pages repository 子路徑。',
  );
}
assert.ok(!existsSync(resolve(outputDirectory, 'api', 'predict')), '靜態輸出不應包含 Server API。');
assert.ok(!existsSync(resolve(outputDirectory, 'api')), '靜態輸出不應包含任何 API 路徑。');
assert.ok(
  existsSync(resolve(outputDirectory, 'images', 'feature_selection_comparison.png')),
  '靜態輸出缺少代表性的分析圖片。',
);
const project1Images = [
  'feature_selection_comparison.png',
  'actual_vs_predicted.png',
  'residual_distribution.png',
  'model_coefficients.png',
  'correlation_heatmap.png',
];

for (const fileName of project1Images) {
  assert.ok(
    existsSync(resolve(outputDirectory, 'images', 'project1', fileName)),
    `靜態輸出缺少 Project 1 圖片：${fileName}`,
  );
}

console.log(
  expectedBasePath
    ? `靜態匯出驗證通過：繁中首頁、${expectedBasePath} 資源子路徑與純靜態輸出皆正確。`
    : '靜態匯出驗證通過：繁中首頁、根目錄資源路徑與純靜態輸出皆正確。',
);
