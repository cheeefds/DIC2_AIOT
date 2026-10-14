import type { Metadata, Viewport } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: '多元線性迴歸雙專題｜50 Startups 與 Boston Housing',
  description: '以多元線性迴歸模型展示 50 Startups 利潤預測與 Boston Housing 住宅價值預測，並整合特徵選擇、模型表現與實驗圖表。',
  keywords: ['機器學習', '多元線性迴歸', '50 Startups', '波士頓房價', 'Kaggle', '特徵選擇', '資料科學'],
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
