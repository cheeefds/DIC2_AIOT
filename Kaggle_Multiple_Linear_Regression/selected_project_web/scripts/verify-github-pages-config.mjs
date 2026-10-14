import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';

const configUrl = pathToFileURL(resolve('next.config.mjs')).href;

function readConfig(environment) {
  const result = spawnSync(
    process.execPath,
    [
      '--input-type=module',
      '--eval',
      `import config from ${JSON.stringify(configUrl)}; process.stdout.write(JSON.stringify({ basePath: config.basePath, assetPrefix: config.assetPrefix }));`,
    ],
    {
      encoding: 'utf8',
      env: {
        ...process.env,
        GITHUB_ACTIONS: environment.GITHUB_ACTIONS ?? '',
        GITHUB_REPOSITORY: environment.GITHUB_REPOSITORY ?? '',
      },
    },
  );

  assert.equal(result.status, 0, result.stderr);
  return JSON.parse(result.stdout);
}

assert.deepEqual(
  readConfig({ GITHUB_ACTIONS: 'true', GITHUB_REPOSITORY: 'student/regression-homework' }),
  { basePath: '/regression-homework', assetPrefix: '/regression-homework' },
  '一般 project repository 必須使用 repository 子路徑。',
);

assert.deepEqual(
  readConfig({ GITHUB_ACTIONS: 'true', GITHUB_REPOSITORY: 'student/student.github.io' }),
  { basePath: '', assetPrefix: '' },
  '使用者首頁 repository 必須部署在網站根目錄。',
);

assert.deepEqual(
  readConfig({ GITHUB_ACTIONS: 'false', GITHUB_REPOSITORY: 'student/regression-homework' }),
  { basePath: '', assetPrefix: '' },
  '非 GitHub Actions 的本機建置不得加入 repository 子路徑。',
);

console.log('GitHub Pages 路徑設定驗證通過：project、使用者首頁與本機模式皆正確。');
