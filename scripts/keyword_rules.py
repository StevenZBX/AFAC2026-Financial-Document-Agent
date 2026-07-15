"""Keyword extraction and domain-specific retrieval rules.

This module intentionally avoids embedding models. It extracts useful lexical
signals from questions/options, including Chinese terms, clause numbers,
amounts, dates, ratios, and domain-specific synonyms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "insurance": [
        "保险责任",
        "身故保险金",
        "现金价值",
        "退保",
        "领取",
        "等待期",
        "免赔额",
        "赔付",
        "保单账户",
        "养老年金",
        "保险金额",
        "保险期间",
        "投保人",
        "被保险人",
        "受益人",
        "豁免",
        "红利",
        "生存金",
        "住院",
        "手术",
        "重大疾病",
        "理赔",
        "续保",
        "犹豫期",
    ],
    "regulatory": [
        "应当",
        "不得",
        "必须",
        "期限",
        "处罚",
        "报告",
        "备案",
        "施行",
        "管理办法",
        "条例",
        "股东大会",
        "董事会",
        "独立董事",
        "信息披露",
        "内幕信息",
        "违规",
        "罚款",
        "警告",
        "吊销",
        "许可证",
        "合规",
        "监管",
        "证监会",
        "银保监",
        "股票代码",
        "上市公司",
        "募集资金",
        "担保",
    ],
    "financial_contracts": [
        "债券",
        "期限",
        "利率",
        "评级",
        "发行人",
        "担保",
        "权利",
        "义务",
        "募集",
        "票面利率",
        "发行规模",
        "兑付",
        "违约",
        "增信",
        "质押",
        "抵押",
        "承销",
        "主承销商",
        "信用评级",
        "到期日",
        "付息",
        "回售",
        "转股",
        "可转债",
        "受托管理人",
        "公司"
    ],
    "financial_reports": [
        "营业收入",
        "净利润",
        "现金流",
        "研发投入",
        "分红",
        "年度",
        "同比",
        "毛利率",
        "归母净利润",
        "扣非净利润",
        "每股收益",
        "资产负债率",
        "总资产",
        "净资产",
        "营业成本",
        "期间费用",
        "存货",
        "应收账款",
        "资本开支",
        "股息",
        "派息",
        "净资产收益率",
        "经营活动",
    ],
    "research": [
        "行业趋势",
        "公司比较",
        "增长率",
        "研究结论",
        "风险提示",
        "市场规模",
        "竞争格局",
        "渗透率",
        "市占率",
        "龙头",
        "头部",
        "产能",
        "出货量",
        "下游",
        "上游",
        "产业链",
        "估值",
        "目标价",
        "买入",
        "推荐",
        "盈利预测",
    ],
}

SYNONYMS: Dict[str, List[str]] = {
    "身故保险金": ["身故", "死亡保险金", "保险金"],
    "现金价值": ["保单现金价值"],
    "退保": ["解除合同", "退保金", "退还"],
    "营业收入": ["营收", "收入"],
    "净利润": ["归母净利润", "利润"],
    "报告": ["报送", "提交", "报备"],
    "期限": ["时限", "期间", "日内", "个月内"],
    "股东大会": ["股东会", "股东大会审议"],
    "董事会": ["董事会决议", "董事会审议"],
    "信息披露": ["披露", "公告", "公开披露"],
    "研发投入": ["研发费用", "研发支出"],
    "现金流": ["现金流量", "经营现金流"],
    "担保": ["保证", "抵押担保", "质押担保"],
    "利率": ["票面利率", "利息"],
    "评级": ["信用评级", "主体评级", "债项评级"],
}

CLAUSE_RE = re.compile(r"第[一二三四五六七八九十百千万0-9]+[章节条款项目]")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:万|万元|亿|亿元|%|％|个工作日|个自然日|日|天|年|个月|月)")
YEAR_RE = re.compile(r"(?:19|20)\d{2}\s*年?")
CODE_RE = re.compile(r"[A-Za-z_]+[_-]?\d{2,}|csrc_\d{4}|strict_v\d+_\d+")
CHINESE_TERM_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9%％]{2,}")

STOPWORDS = {
    "关于",
    "以下",
    "下列",
    "正确",
    "错误",
    "的是",
    "根据",
    "结合",
    "假设",
    "分别",
    "需要",
    "应当",
    "进行",
    "其中",
}

@dataclass(frozen=True)
class QuerySignals:
    """Structured query signals used by keyword and BM25 retrieval."""

    keywords: List[str]
    clauses: List[str]
    numbers: List[str]
    dates: List[str]
    codes: List[str]

    def all_terms(self) -> List[str]:
        return unique_preserve_order(
            [*self.keywords, *self.clauses, *self.numbers, *self.dates, *self.codes]
        )


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        item = normalize_text(item)
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output
def normalize_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()

def extract_signals(text: str, domain: str | None = None) -> QuerySignals:
    text = normalize_text(text)
    clauses = CLAUSE_RE.findall(text)
    numbers = NUMBER_RE.findall(text)
    dates = YEAR_RE.findall(text)
    codes = CODE_RE.findall(text)

    candidates = CHINESE_TERM_RE.findall(text)
    keywords = [x for x in candidates if x not in STOPWORDS and len(x) >= 2]
    if domain:
        for term in DOMAIN_KEYWORDS.get(domain, []):
            if term in text:
                keywords.append(term)
        for term in DOMAIN_KEYWORDS.get(domain, []):
            if any(part in text for part in term.split()):
                keywords.append(term)

    expanded: List[str] = []
    for keyword in keywords:
        expanded.append(keyword)
        expanded.extend(SYNONYMS.get(keyword, []))

    return QuerySignals(
        keywords=unique_preserve_order(expanded),
        clauses=unique_preserve_order(clauses),
        numbers=unique_preserve_order(numbers),
        dates=unique_preserve_order(dates),
        codes=unique_preserve_order(codes),
    )
def build_question_query(question: dict, include_options: bool = True) -> str:
    parts = [question.get("question", "")]
    if include_options:
        options = question.get("options") or {}
        if isinstance(options, dict):
            parts.extend(str(value) for value in options.values())
    return " ".join(normalize_text(part) for part in parts if part)


def option_queries(question: dict) -> Dict[str, str]:
    base = normalize_text(question.get("question", ""))
    options = question.get("options") or {}
    if not isinstance(options, dict):
        return {}
    return {label: f"{base} {normalize_text(text)}" for label, text in options.items()}


def keyword_score(text: str, signals: QuerySignals, domain: str | None = None) -> float:
    text = normalize_text(text)
    if not text:
        return 0.0

    score = 0.0
    for term in signals.keywords:
        if term and term in text:
            score += 1.0 + min(len(term), 8) * 0.05
    for term in signals.clauses:
        if term in text:
            score += 3.0
    for term in signals.numbers:
        if term.replace(" ", "") in text.replace(" ", ""):
            score += 2.5
    for term in signals.dates:
        if term in text:
            score += 1.5
    for term in signals.codes:
        if term in text:
            score += 2.0
    for term in DOMAIN_KEYWORDS.get(domain or "", []):
        if term in text:
            score += 0.3
    return score


def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize Chinese/English text with deterministic lightweight rules."""

    text = normalize_text(text).lower()
    coarse = CHINESE_TERM_RE.findall(text)
    tokens: List[str] = []
    for item in coarse:
        tokens.append(item)
        if re.search(r"[\u4e00-\u9fff]", item) and len(item) > 2:
            tokens.extend(item[i : i + 2] for i in range(len(item) - 1))
            tokens.extend(item[i : i + 3] for i in range(len(item) - 2))
    return [token for token in tokens if token and token not in STOPWORDS]


def query_terms(text: str, domain: str | None = None) -> List[str]:
    signals = extract_signals(text, domain)
    return unique_preserve_order([*signals.all_terms(), *tokenize_for_bm25(text)])






