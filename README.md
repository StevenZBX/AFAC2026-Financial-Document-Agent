# 全量 Markdown 处理结果

此目录使用与既有流程一致的思路，批量处理 MinerU 生成的 Markdown：按标题层级分段、按长度切块、保留前文重叠，并生成可用于检索的 `retrieval_text`。

## 目录

- `code/process_all_markdown.py`：通用递归处理脚本
- `code/run_all.sh`：一键重跑入口
- `output/<类别>/documents/*.json`：逐文档结果
- `output/<类别>/document_index.json`：文档索引
- `output/<类别>/process_data.json`：类别完整数据
- `output/summary.json`：全量统计

## 重跑

```bash
sh code/run_all.sh
```

默认输入目录为 `/Users/limeixuan/Desktop/mineru_otuput`。
