# velocity V1 最小 CI 与依赖分层说明

## 1. 文档定位

本文档用于 velocity V1 Step 9：CI 收口与最小验收，说明最小 CI、requirements 分层、GitHub Actions 工作流、最小 pytest 阻断集，以及 extended / optional CI 的边界。

本文档只说明 Step 9 已确认的最小 CI 执行口径，不重写 velocity 模块设计，不扩展到 Step 10，不回退修改 Step 1–Step 8 已收口成果。

## 2. 最小 CI 定位

velocity V1 最小 CI 的目标是建立一个轻量、稳定、可复现的阻断性 pytest 入口，用于尽早发现以下退化：

- velocity contract / schema 退化；
- P/S 双相速度模型构建链路退化；
- runtime 读取与使用速度模型能力退化；
- `traveltime_ready` 关键交付链路退化。

最小 CI 不承担完整 QC 图件检查、overview 图形检查、optional 依赖完整性检查、大数据样例回归或人工验收替代职责。

## 3. GitHub Actions 最小 CI 工作流位置

最小 CI 工作流文件位置为：

```text
.github/workflows/velocity-minimal-ci.yml
```

该 workflow 固定使用 Python 3.10，并在 YAML 中显式写明 pytest 命令，不依赖 `pytest.ini`、`tox.ini`、`noxfile.py` 或 `pyproject.toml` 中的隐式测试配置。

## 4. requirements 分层说明

Step 9 采用以下 requirements 分层文件：

```text
requirements.txt
requirements-test.txt
requirements-dev.txt
requirements-optional.txt
```

### 4.1 requirements.txt

`requirements.txt` 是 velocity 最小运行 / 构建 / contract / runtime 基础依赖层。该文件保持轻量，不放入 pytest、matplotlib、notebook、jupyter、ipykernel 或其他可选重型依赖。

当前用途：

- 支撑 velocity 基础构建链路；
- 支撑 contract / schema 基础校验；
- 支撑 runtime 与 `traveltime_ready` 最小测试路径；
- 作为 `requirements-test.txt` 的基础依赖层。

### 4.2 requirements-test.txt

`requirements-test.txt` 是最小 CI pytest 阻断集的安装入口。该文件引用 `requirements.txt`，并额外安装 pytest。

最小 GitHub Actions CI 应只安装该文件，不安装 optional 层或 dev 层。

### 4.3 requirements-dev.txt

`requirements-dev.txt` 是开发辅助依赖入口，不作为最小 CI 的强制安装入口。

当前原则是：在未形成统一开发工具规范前，不主动引入 black、ruff、mypy、pytest-cov 等开发工具，避免把未确认的开发规范混入 Step 9 最小 CI 阻断路径。

### 4.4 requirements-optional.txt

`requirements-optional.txt` 用于 optional / extended CI、图形 QC、overview 或其他非阻断性扩展验收路径。

当前 `matplotlib` 放入该层，不进入 `requirements-test.txt`。

## 5. matplotlib 不进入最小 CI 的原因

`matplotlib` 主要服务于 QC 图件、overview 图形和扩展可视化验收。Step 9 最小 CI 的目标是快速验证 contract、P/S 构建、runtime 与 `traveltime_ready` 关键链路，不应因图形依赖、后端配置或可视化输出引入额外不稳定因素。

因此：

- `matplotlib` 放入 `requirements-optional.txt`；
- 最小 CI 不安装 `requirements-optional.txt`；
- 涉及图形输出的 QC 测试进入 extended / optional CI。

## 6. 最小 CI pytest 阻断集

最小 CI 阻断集显式运行以下测试文件：

```cmd
python -m pytest tests\velocity\test_velocity_contract_schema.py tests\velocity\test_build_velocity.py tests\velocity\test_runtime_velocity_model.py tests\velocity\test_traveltime_ready.py -q -s --basetemp=.pytest_tmp
```

在 GitHub Actions YAML 中使用等价的跨平台路径写法。

覆盖关系如下：

- `tests/velocity/test_velocity_contract_schema.py`：覆盖 velocity contract / schema 基础约束；
- `tests/velocity/test_build_velocity.py`：覆盖 P/S 双相速度模型构建与核心交付产物；
- `tests/velocity/test_runtime_velocity_model.py`：覆盖 runtime 读取与模型使用能力；
- `tests/velocity/test_traveltime_ready.py`：覆盖 `traveltime_ready` 关键交付链路。

## 7. test_qc_velocity.py 的边界

`tests/velocity/test_qc_velocity.py` 不进入最小 CI 阻断集，正式放入 extended / optional CI 候选范围。

原因是该测试覆盖 velocity QC 聚合链路，可能涉及 artifact QC、layerize QC、summary QC、overview 或图件相关路径。该类测试对完整工程验收有价值，但不应污染最小 CI 阻断集。

## 8. extended / optional CI 边界

extended / optional CI 可用于覆盖以下内容：

- `tests/velocity/test_qc_velocity.py`；
- 需要 `requirements-optional.txt` 的 QC / 图形 / overview 路径；
- velocity QC summary、artifact QC、layerize QC、图件弱断言；
- 后续非阻断性扩展验收。

extended / optional CI 的设计原则是：可以增强验收覆盖，但不得反向增加最小 CI 的硬依赖，不得把 optional 重型依赖写入 `requirements-test.txt`。

## 9. 本地最小验证命令

在仓库根目录执行：

```cmd
python -m pip install -r requirements-test.txt
python -m pytest tests\velocity\test_velocity_contract_schema.py tests\velocity\test_build_velocity.py tests\velocity\test_runtime_velocity_model.py tests\velocity\test_traveltime_ready.py -q -s --basetemp=.pytest_tmp
```

该命令只验证最小 CI 阻断集，不等同于完整 QC 验收，也不等同于远端 GitHub Actions 已通过。

## 10. 最小 CI 不依赖大数据与可选重型库的说明

最小 CI 仅使用仓库内 `tests/velocity/` 对应轻量测试路径与最小测试资产，不引入外部大数据下载、人工数据目录、个人机器绝对路径或可选图形依赖。

如后续需要增加大数据样例、完整 QC 图件、overview 或跨模块扩展验收，应进入 extended / optional CI，而不是修改最小 CI 阻断集。

## 11. P/S 不退化的最小保障

P/S 不退化由最小 CI 中的 build 与 contract 测试共同承担基础保护：

- contract / schema 测试约束 velocity layer table 的字段、边界与速度值合法性；
- build 测试覆盖 P-wave 与 S-wave 速度字段承接、构建输出与代表性产物生成；
- runtime 测试补充模型读取与运行时使用能力保护。

该保护属于最小保障，不替代后续完整物理合理性审查或工程级多样例回归。

## 12. traveltime_ready 不退化的最小保障

`traveltime_ready` 不退化由最小 CI 中的 `tests/velocity/test_traveltime_ready.py` 承担基础保护，并与 build/runtime 测试形成前后链路约束。

最小保障关注：

- `traveltime_ready` 相关目录或代表性产物能够生成；
- velocity 输出能够保持面向 traveltime 的基础承接能力；
- Step 8 已形成的 `traveltime_ready` 主线不因 Step 9 的 CI / requirements / docs 修改而退化。

## 13. 后续扩展原则

后续如需扩展 CI，应遵守以下原则：

- 最小 CI 继续保持轻量、短小、稳定；
- optional / extended CI 承接 QC、图形、overview、大样例和非阻断扩展检查；
- 不把 `requirements-optional.txt` 反向合并到 `requirements-test.txt`；
- 不把 `test_qc_velocity.py` 反向并入最小 CI 阻断集，除非经后续方案确认并同步调整依赖分层；
- 不通过新增 pytest 隐式配置改变最小 CI 的执行边界。
