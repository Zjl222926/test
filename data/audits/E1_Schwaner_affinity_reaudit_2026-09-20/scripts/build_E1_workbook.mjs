import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const dir = 'C:/Users/admin/Desktop/新/outputs/E1_Schwaner_affinity_reaudit_2026-09-20';
const payload = JSON.parse(await fs.readFile(path.join(dir, 'derived_data/workbook_payload.json'), 'utf8'));
const wb = Workbook.create();
const ink = '#243546';
const dark = '#254663';
const sea = '#287C83';
const pale = '#EAF3F3';
const amber = '#F4E4BC';
const font = 'Arial';

function colName(index) {
  let out = '';
  for (let n = index + 1; n > 0; n = Math.floor((n - 1) / 26)) out = String.fromCharCode(65 + ((n - 1) % 26)) + out;
  return out;
}

function addTable(name, key, subtitle, options = {}) {
  const { columns, rows } = payload[key];
  const sh = wb.worksheets.add(name);
  sh.showGridLines = false;
  const end = colName(columns.length - 1);
  sh.getRange('A1').values = [[name]];
  sh.getRange('A1').format.font = { name: font, size: 14, bold: true, color: dark };
  sh.getRange(`A2:${end}2`).format.font = { name: font, size: 10, italic: true, color: '#5D6C73' };
  sh.getRange('A2').values = [[subtitle]];
  sh.getRange(`A4:${end}4`).values = [columns];
  sh.getRange(`A4:${end}4`).format = {
    fill: dark,
    font: { name: font, size: 10, bold: true, color: '#FFFFFF' },
    rowHeight: 28,
    verticalAlignment: 'center',
  };
  if (rows.length) {
    for (let start = 0; start < rows.length; start += 100) {
      const chunk = rows.slice(start, start + 100);
      sh.getRangeByIndexes(4 + start, 0, chunk.length, columns.length).values = chunk;
    }
    sh.getRangeByIndexes(4, 0, rows.length, columns.length).format = {
      font: { name: font, size: 10, color: ink },
      rowHeight: options.longRows ? 44 : 25,
      verticalAlignment: 'center',
    };
    if (options.longRows) {
      for (const c of ['formation', 'formation_audit', 'audit_reason', 'source_identity_note']) {
        const ix = columns.indexOf(c);
        if (ix >= 0) sh.getRangeByIndexes(4, ix, rows.length, 1).format.wrapText = true;
      }
    }
    const table = sh.tables.add(`A4:${end}${4 + rows.length}`, true, `T${name.replace(/[^A-Za-z0-9]/g, '')}`);
    table.showFilterButton = true;
  }
  sh.freezePanes.freezeRows(4);
  for (let i = 0; i < columns.length; i++) {
    const c = columns[i];
    let width = 20;
    if (c === 'sample_id') width = 17;
    else if (c === 'sample_instance_uid') width = 43;
    else if (c === 'study_id') width = 33;
    else if (c === 'doi' || c.endsWith('_url')) width = 47;
    else if (c === 'audit_reason' || c === 'source_identity_note') width = 74;
    else if (c === 'retained_sample_ids') width = 80;
    else if (c.includes('formation')) width = 48;
    else if (c === 'material_group' || c === 'source_component') width = 31;
    else if (c.includes('age_rows') || c.includes('samples') || c.includes('accepted') || c === 'n_valid_ages') width = 20;
    sh.getRangeByIndexes(0, i, 1, 1).format.columnWidth = width;
  }
  for (const numeric of ['latitude', 'longitude']) {
    const ix = columns.indexOf(numeric);
    if (ix >= 0 && rows.length) sh.getRangeByIndexes(4, ix, rows.length, 1).setNumberFormat('0.00000');
  }
  for (const numeric of ['n_valid_ages', 'n_author_accepted_known', 'original_candidate_samples', 'retained_samples', 'external_samples', 'original_valid_age_rows', 'retained_valid_age_rows']) {
    const ix = columns.indexOf(numeric);
    if (ix >= 0 && rows.length) sh.getRangeByIndexes(4, ix, rows.length, 1).setNumberFormat('#,##0');
  }
  if (options.tabColor) sh.tabColor = options.tabColor;
  return sh;
}

const summary = wb.worksheets.add('Summary');
summary.showGridLines = false;
summary.tabColor = dark;
summary.getRange('A2').values = [['E1 碎屑训练端元地质复审']];
summary.getRange('A2').format.font = { name: font, size: 15, bold: true, color: dark };
summary.getRange('A3').values = [['定义：SW Borneo / Schwaner affinity；不做 core/expanded 分级，也不设颗粒数门槛']];
summary.getRange('A3').format.font = { name: font, size: 10, italic: true, color: '#5D6C73' };
summary.getRange('A5:D5').values = [['样品数', '独立 DOI/study', 'U–Pb 有效年龄行', '作者接受规则校正后*']];
summary.getRange('A5:D5').format = { fill: dark, font: { name: font, size: 10, bold: true, color: '#FFFFFF' }, rowHeight: 27 };
summary.getRange('A6:D6').values = [[
  payload.summary.retained_samples,
  payload.summary.retained_studies,
  payload.summary.retained_grains_valid_age,
  payload.summary.retained_grains_with_breitfeld_author_accept_only,
]];
summary.getRange('A6:D6').format = { fill: pale, font: { name: font, size: 13, bold: true, color: sea }, rowHeight: 30 };
summary.getRange('A6:D6').setNumberFormat('#,##0');
summary.getRange('A8:E8').values = [['统计口径', '样品/记录数', '独立 DOI', 'U–Pb 年龄行', '说明']];
summary.getRange('A8:E8').format = { fill: dark, font: { name: font, size: 10, bold: true, color: '#FFFFFF' }, rowHeight: 27 };
summary.getRange('A9:E12').values = payload.comparison.rows;
summary.getRange('A9:E12').format = { font: { name: font, size: 10, color: ink }, rowHeight: 28 };
summary.getRange('B9:D12').setNumberFormat('#,##0');
summary.getRange('A14').values = [['判定说明']];
summary.getRange('A14').format.font = { name: font, size: 11, bold: true, color: dark };
summary.getRange('A15').values = [['保留合理区域供源与再循环样品；9 个非目标地质体系样品移至外部参照，原数据库和 E 标签均未改动。']];
summary.getRange('A16').values = [['Breitfeld 2020 正文称 8 个河砂，但原始附表列出 9 个不同编号/坐标且合计 1,025 次分析；按附表身份保留 9 个。']];
summary.getRange('A17').values = [['STB74b 在原库地层空白；原文 7.2.3 明确为 Bako–Mintu Sandstone，衍生表纠正，非 Ngili Sandstone。']];
summary.getRange('A18').values = [['* 2,986 仅将 Breitfeld 2020 的 1,025 行替换为原文 accept=Y 的 837 行；其他研究未作跨研究统一谐和度再筛选。']];
summary.getRange('A19').values = [['TARGET_OVERLAP_REFERENCE 19 行及 AUX_BEDROCK_REFERENCE 148 行从未进入本次碎屑训练候选。']];
summary.getRange('A15:A19').format.font = { name: font, size: 10, color: ink };
summary.getRange('A18:E18').format.fill = amber;
summary.getRange('A1:A20').format.columnWidth = 77;
summary.getRange('B1:D20').format.columnWidth = 22;
summary.getRange('D1:D20').format.columnWidth = 27;
summary.getRange('E1:E20').format.columnWidth = 82;

addTable('Retained E1', 'retained', '30 个物理碎屑样品；原数据库字段另见完整 CSV；地层纠正仅写入审计列', { tabColor: sea, longRows: true });
addTable('External reference', 'external', '9 个样品不进入 E1 训练；均保留在原数据库和外部参照 CSV', { tabColor: '#B87644', longRows: true });
addTable('Studies', 'studies', '原候选与统一 E1 的逐 DOI 样品数、颗粒数对照');
addTable('River source IDs', 'river_source', 'Breitfeld 2020 原始 Supplementary Table 3 的 9 个物理编号与坐标');

wb.recalculate();
for (const [name, range, file] of [
  ['Summary', 'A2:E19', 'preview_summary.png'],
  ['Retained E1', 'A1:J9', 'preview_retained.png'],
  ['External reference', 'A1:J9', 'preview_external.png'],
  ['Studies', 'A1:H12', 'preview_studies.png'],
  ['River source IDs', 'A1:B13', 'preview_river.png'],
]) {
  const preview = await wb.render({ sheetName: name, range, scale: 1.3, format: 'png' });
  await fs.writeFile(path.join(dir, 'derived_data', file), new Uint8Array(await preview.arrayBuffer()));
}
const check = await wb.inspect({ kind: 'table', range: 'Summary!A5:E12', include: 'values,formulas', tableMaxRows: 12, tableMaxCols: 5 });
await fs.writeFile(path.join(dir, 'derived_data', 'workbook_inspect.ndjson'), check.ndjson, 'utf8');
const errors = await wb.inspect({
  kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',
  options: { useRegex: true, maxResults: 300 }, summary: 'final formula error scan',
});
await fs.writeFile(path.join(dir, 'derived_data', 'workbook_errors.ndjson'), errors.ndjson, 'utf8');
const out = await SpreadsheetFile.exportXlsx(wb);
await out.save(path.join(dir, 'E1_Schwaner_affinity_reaudit.xlsx'));
console.log('Saved E1 audit workbook');
