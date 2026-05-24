# ML 持续学习 × 数据版本控制 Demo：lakeFS vs MatrixOne git4data

模拟「机器学习数据持续流入 + 模型持续学习」的场景，在 **lakeFS** 和 **MatrixOne
git4data** 两套数据版本控制方案上跑**完全相同**的五幕剧情，对比能力差异与适应性。

结论与详细分析见 **[COMPARISON.md](COMPARISON.md)**。

## 五幕剧情（两套实现一一对应）

| 幕 | MLOps 诉求 | MatrixOne | lakeFS |
|---|---|---|---|
| 1 | 数据分批流入 + 增量学习 | 每批 `CREATE SNAPSHOT` | 每批 `commit` + `tag` |
| 2 | 两版本间发生了什么 | 原生 `DATA BRANCH DIFF`（行级） | 原生 `diff`（对象级） |
| 3 | 复现历史模型 | `SELECT … {snapshot=}` 重训 | 读 `ref` 下对象重训 |
| 4 | 坏批次数据回滚 | `RESTORE … FROM SNAPSHOT` | 原生 `revert` |
| 5 | 分支清洗实验 + 合并 | 原生 `DATA BRANCH CREATE/DIFF/PICK/MERGE`（行级 + cherry-pick） | 原生 `branch` + `merge`（对象级） |

> 两套都用各自的**原生** git 命令。关键差异：MatrixOne 的 diff/merge 是**行级**
> 还支持 cherry-pick，lakeFS 是**对象/整文件级**但作用于**整个仓库**。详见 COMPARISON.md。

两套共用同一数据生成器与同一增量模型，因此精度数字跨系统一致：

```
ACT1 干净流入: 0.977 → 0.989    ACT3 复现 v3: 0.970 (reproducible == True)
ACT4 投毒→回滚: 0.886 → 0.989   ACT5 清洗实验: 0.989 → 0.9955 (修正 310 行错标)
```

## 目录结构

```
common/          共享：合成数据流(data_stream) + 增量模型(model, SGDClassifier.partial_fit)
config.py        领域常量 + 从 .mo.cnf 读 MatrixOne 连接
matrixone/       MatrixOne git4data 实现（mo_client / git4data / repo / run_demo）
lakefs_demo/     lakeFS 实现（lk_config / start_lakefs.sh / run_demo）
experiments/     深度实验脚本（见下）
COMPARISON.md    能力对比与选型报告（§5 为深度实验实测数据）
```

## 深度实验（`experiments/`，均实跑于 v3.0.11、自清理）

| 脚本 | 验证点 | 依赖 |
|---|---|---|
| `exp_scale.py` | 1万→100万行下 snapshot/branch/clone/restore 恒定耗时（零拷贝），diff ∝ 变更量 | 仅 MatrixOne |
| `exp_incremental_diff.py` | 持续学习只训练 delta：用 `DATA BRANCH DIFF` 取变更行，每轮成本 ∝ 变更量而非总量 | 仅 MatrixOne |
| `exp_concurrent_merge.py` | 并发标注分支合并冲突：`WHEN CONFLICT FAIL/SKIP/ACCEPT` 行级语义 | 仅 MatrixOne |
| `exp_branch_advanced.py` | schema 演进会打断 diff/merge；库级快照/恢复多表原子一致 | 仅 MatrixOne |
| `exp_stage_datalink.py` | STAGE+datalink+load_file 编目 OSS 文件；快照只版本"引用"非"字节"；DIFF OUTPUT FILE 导出 SQL 补丁 | MatrixOne + OSS(.lakefs.env) |
| `exp_sft_curation.py` | SFT 数据策展：去重/质量过滤/去污染全在版本化表上原地 SQL，DIFF 给行级溯源 | 仅 MatrixOne |
| `exp_rlhf_preference.py` | RLHF 偏好数据：SQL 算标注一致性/共识，cherry-pick 改判，按 reward-model 运行 pin 快照 | 仅 MatrixOne |
| `exp_bench_curation.py` | 正面基准：同一份 SFT 策展，MatrixOne 原地 SQL vs lakeFS+DuckDB 读/算/写回（warmup+中位数） | MatrixOne + OSS + DuckDB |
| `exp_continuous_annotation.py` | 场景①持续标注：训练快照 + DIFF 算两次训练差异 + PITR 恢复任意未打标时刻 | 仅 MatrixOne |
| `exp_stream_versioning.py` | 场景④实时流：每条 INSERT 一个事务版本，PITR 重建任意微秒的训练集 | 仅 MatrixOne |
| `exp_write_audit_publish.py` | 场景③WAP：写暂存分支→SQL 质量门禁审计→原子 MERGE 发布 | 仅 MatrixOne |
| `exp_multimodal_catalog.py` | 场景②多模态：datalink 编目 + vecf32 近重检测 + git4data 版本化目录（字节交 lakeFS） | MatrixOne + OSS |
| `exp_integration_poc.py` | **集成 PoC**：lakeFS 管字节版本 + MatrixOne 目录 pin lakeFS commit → 字节级时间旅行 + 行级"改了啥" + 可直读 URL | lakeFS + MatrixOne + OSS |
| `exp_agent_evolution.py` | **非 ML**：agent trace+大脑存为版本化表，branching 做进化（学习→对比→MERGE，成功率 50%→100%），坏变异 RESTORE 回滚，冲突/cherry-pick | 仅 MatrixOne |

```bash
python3 -m experiments.exp_scale
python3 -m experiments.exp_incremental_diff
python3 -m experiments.exp_concurrent_merge
python3 -m experiments.exp_branch_advanced
python3 -m experiments.exp_stage_datalink   # 需 .lakefs.env 里的 OSS 凭证
python3 -m experiments.exp_sft_curation     # SFT 数据策展
python3 -m experiments.exp_rlhf_preference  # RLHF 偏好数据策展
python3 -m experiments.exp_bench_curation   # 性能基准 (需 OSS + duckdb)
python3 -m experiments.exp_continuous_annotation  # 场景① 持续标注 + PITR
python3 -m experiments.exp_stream_versioning      # 场景④ 实时流版本化
python3 -m experiments.exp_write_audit_publish    # 场景③ Write-Audit-Publish
python3 -m experiments.exp_multimodal_catalog     # 场景② 多模态目录 (需 OSS)
python3 -m experiments.exp_integration_poc        # lakeFS+MatrixOne 集成 PoC (需 OSS+lakeFS)
python3 -m experiments.exp_agent_evolution        # 非 ML：agent trace + branching 进化
```

> 场景匹配度（COMPARISON.md §9）：持续标注 + 实时流 **MatrixOne 优势显著**（每事务版本 +
> PITR 任意时间点 + 行级 DIFF）；WAP 两者都行；多模态字节是 **lakeFS 主场**（MatrixOne 管目录侧）。

> 基准结论（§7）很诚实：原始策展吞吐 **lakeFS+DuckDB 反而更快**（DuckDB 极快、parquet 小、
> 远程 MatrixOne 有网络往返）。MatrixOne 的价值在**集成度/行级语义/可复现/少运维**，不是裸算速度。

## 运行

### 前置
```bash
pip3 install -r requirements.txt
```

### MatrixOne demo（开箱即用，直连云实例）
连接信息放在 `.mo.cnf`（已 gitignore）：
```ini
[client]
host=<matrixone-host>
port=6001
user=<account>:<user>:<role>
password=<password>
```
运行（幂等，可重复跑）：
```bash
python3 -m matrixone.run_demo
```
脚本只在自建的 `ml_git4data_demo` 库中操作，不触碰其它库。

### lakeFS demo（需 Docker + 对象存储）
1. 启动 Docker Desktop。
2. 复制 `.lakefs.env.example` → `.lakefs.env`，填入阿里云 OSS 桶/endpoint/AK/SK
   （`.lakefs.env` 已 gitignore）。
3. 启动本地 lakeFS server（元数据走本地，blockstore 指向 OSS，监听 127.0.0.1:8200）：
   ```bash
   bash lakefs_demo/start_lakefs.sh
   ```
4. 运行 demo（幂等：每次先清空 OSS 命名空间前缀再重建 repo）：
   ```bash
   python3 -m lakefs_demo.run_demo
   ```

> lakeFS 架构：常驻 **server**（git 语义）+ **OSS blockstore**（存数据）+ 本地元数据
> KV + **客户端 SDK**。「云上 OSS + 客户端」之外还需要这个 server 进程。

### lakeFS 实跑踩过的坑（已在代码/脚本中处理）
- **端口冲突 / IPv4-IPv6**：lakeFS 改用 `127.0.0.1:8200`，并显式绑 IPv4，避免与本机
  其它占用 `:8000` 的进程冲突（曾出现 curl 走 IPv6 通、Python 走 IPv4 命中别的服务报 404）。
- **OSS 必须虚拟主机风格寻址**：lakeFS 侧 `FORCE_PATH_STYLE=false`；boto3 清理命名空间
  时用 `addressing_style='virtual'`。
- **OSS 批量删除需 Content-MD5**：清理命名空间改用单对象 `DeleteObject`。
- **OSS AccessKey 必须启用**，否则报 `InvalidAccessKeyId: disabled`。
- 详见 [COMPARISON.md](COMPARISON.md) §3。

## 安全说明
`.mo.cnf` / `.lakefs.env` 含明文凭证，已在 `.gitignore` 中排除，不要提交。
