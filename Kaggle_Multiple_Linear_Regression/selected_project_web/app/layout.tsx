import type { Metadata, Viewport } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: '波士頓住宅價值預測｜多元線性迴歸儀表板',
  description: '以多元線性迴歸模型進行即時住宅價值預測，並展示特徵選擇、模型表現與實驗圖表。',
  keywords: ['機器學習', '多元線性迴歸', '波士頓房價', 'Kaggle', '特徵選擇', '資料科學'],
  authors: [{ name: 'Data Intelligence Team' }],
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-Hant">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
