import { parse } from '@std/csv/parse';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

// 서울시에서 운영하는 공공자전거(따릉이)에 대한 기간별 현황 데이터입니다.
// https://data.seoul.go.kr/dataList/OA-14994/F/1/datasetView.do
const [, ...rows] = parse(readFileSync(join(import.meta.dirname, 'data.csv'), 'utf-8'));

rows.forEach(([yyyyMMdd, value], index) => {
	if (
		!/^\d{4}-\d{2}-\d{2}$/.test(yyyyMMdd) || //
		!/^\d*$/.test(value)
	)
		throw new Error(yyyyMMdd);

	if (index === 0) return;

	const a = new Date(yyyyMMdd).valueOf();
	const b = new Date(rows[index - 1][0]).valueOf();
	const day = 24 * 60 * 60 * 1000;
	if (a - b !== day) throw new Error(yyyyMMdd);
});
