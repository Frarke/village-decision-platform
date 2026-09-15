"""生成经真实计算得到的离线备用成果与三随机种子算法验证记录。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from model.benchmarks import validate_standard_functions
import pandas as pd
from model.i18n import chinese_frame
from model.nsga2 import solve_nsga2
from model.schemas import SolverConfig, load_measures, load_units
from model.solver import parameters_and_assumptions, result_to_json, solve_exact
from model.validation import validate_runs
from model.spatial import load_topology


ROOT = Path(__file__).resolve().parent


def main() -> None:
    output = ROOT / "sample_output"
    output.mkdir(exist_ok=True)
    units = load_units(ROOT / "data/default_units.csv")
    measures = load_measures(ROOT / "data/default_measures.csv")
    config = SolverConfig()
    topology = load_topology(ROOT / "data/spatial_topology.json")
    exact = solve_exact(units, measures, config, ROOT / "data/assumptions.csv", topology=topology)
    nsga_runs = [
        solve_nsga2(units, measures, replace(config, random_seed=seed), ROOT / "data/assumptions.csv", topology=topology)
        for seed in (42, 123, 2026)
    ]
    nsga = nsga_runs[0]
    validation = validate_runs(exact, nsga_runs)
    benchmark_frames = []
    benchmark_examples = {}
    for function_name in ("ZDT1", "ZDT2", "ZDT3", "DTLZ2"):
        frame, examples = validate_standard_functions(functions=(function_name,))
        benchmark_frames.append(frame); benchmark_examples.update(examples)
        print(f"标准函数{function_name}三随机种子验证完成。", flush=True)
    benchmark_summary = pd.concat(benchmark_frames, ignore_index=True)
    exports = {
        "01_治理单元及标签.csv": chinese_frame(exact.tagged_units),
        "02_工程措施库.csv": chinese_frame(exact.measure_library),
        "03_准入筛选结果.csv": chinese_frame(exact.screening),
        "04_精确枚举可行方案.csv": chinese_frame(exact.all_feasible),
        "05_精确Pareto方案.csv": chinese_frame(exact.pareto),
        "06_NSGA-II可行方案.csv": chinese_frame(nsga.all_feasible),
        "07_NSGA-II-Pareto方案.csv": chinese_frame(nsga.pareto),
        "08_代表方案.csv": chinese_frame(exact.representatives),
        "09_代表方案单元明细.csv": chinese_frame(exact.representative_details),
        "10_NSGA-II迭代历史.csv": chinese_frame(nsga.iteration_history),
        "11_算法验证指标.csv": validation.metrics,
        "12_多随机种子验证汇总.csv": validation.seed_summary,
        "13_当前模型参数与依据状态.csv": parameters_and_assumptions(exact),
        "17_标准函数多随机种子验证.csv": benchmark_summary,
    }
    for filename, frame in exports.items():
        frame.to_csv(output / filename, index=False, encoding="utf-8-sig")
    (output / "14_精确枚举完整结果.json").write_text(result_to_json(exact), encoding="utf-8")
    (output / "15_NSGA-II完整结果.json").write_text(result_to_json(nsga), encoding="utf-8")
    (output / "16_算法验证结论.txt").write_text(validation.conclusion, encoding="utf-8")
    benchmark_summary.groupby("标准函数")[["IGD", "GD", "Spacing", "Hypervolume"]].agg(["mean", "std"]).to_csv(
        output / "18_标准函数指标均值与波动.csv", encoding="utf-8-sig"
    )
    for name, result in benchmark_examples.items():
        result.history.to_csv(output / f"19_{name}_收敛历史.csv", index=False, encoding="utf-8-sig")
    primary = validation.seed_summary.iloc[0]
    print(
        f"精确枚举：{exact.run_metadata['feasible_plans']}个可行方案，{exact.run_metadata['pareto_plans']}个Pareto方案；"
        f"NSGA-II：{nsga.run_metadata['feasible_plans']}个已搜索可行方案，{nsga.run_metadata['pareto_plans']}个非支配方案；"
        f"完全方案覆盖率{primary['精确Pareto集完全方案覆盖率']:.1%}，近似覆盖率{primary['目标空间近似覆盖率']:.1%}。"
    )


if __name__ == "__main__":
    main()
