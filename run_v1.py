"""完整流程和小规模对照：python run_v1.py --compare。"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from model.data_adapter import grid_features
from model.nsga2 import solve_nsga2
from model.result_exporter import export_files, export_zip
from model.service import load_sample_request, model_context, optimize_request
from model.solver import solve_exact
from model.validation import validate_runs


def main():
    parser = argparse.ArgumentParser(description="村域空间综合整治完整计算与成果导出")
    parser.add_argument("--input", type=Path, help="完整项目JSON；省略时使用明确标记的模拟空间数据")
    parser.add_argument("--output", type=Path, default=Path("output/v1"))
    parser.add_argument("--compare", action="store_true", help="实际运行精确枚举和三个种子NSGA-II对照")
    args = parser.parse_args()
    request = json.loads(args.input.read_text("utf-8-sig")) if args.input else load_sample_request()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(args.output / "run.log", encoding="utf-8"), logging.StreamHandler()])
    result = optimize_request(request)
    for name, value in export_files(result).items():
        (args.output / name).write_bytes(value)
    (args.output / "input_request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), "utf-8")
    (args.output / "decision_results.zip").write_bytes(export_zip(result))
    if args.compare and result["status"] == "completed":
        _, units, measures, config, topology, _, _ = model_context(request)
        exact = solve_exact(units, measures, config, topology=topology)
        nsga_runs = [solve_nsga2(units, measures, replace(config, random_seed=seed), topology=topology)
                     for seed in (42, 123, 2026)]
        validation = validate_runs(exact, nsga_runs)
        validation.seed_summary.to_csv(args.output / "solver_comparison.csv", index=False, encoding="utf-8-sig")
        validation.metrics.to_csv(args.output / "validation_metrics.csv", index=False, encoding="utf-8-sig")
        exact.pareto.to_csv(args.output / "exact_pareto.csv", index=False, encoding="utf-8-sig")
        exact.all_feasible.to_csv(args.output / "exact_feasible.csv", index=False, encoding="utf-8-sig")
        for run in nsga_runs:
            run.iteration_history.to_csv(args.output / f"history_seed_{run.run_metadata['random_seed']}.csv", index=False, encoding="utf-8-sig")
        comparisons = {"exact": exact.run_metadata, "nsga_runs": [run.run_metadata for run in nsga_runs],
                       "conclusion": validation.conclusion, "seeds": validation.seed_summary.to_dict("records")}
        (args.output / "validation.json").write_text(json.dumps(comparisons, ensure_ascii=False, indent=2), "utf-8")
        logging.info(validation.conclusion)
        logging.info("精确枚举组合%d；可行%d；非劣%d", exact.run_metadata["estimated_combinations"], len(exact.all_feasible), len(exact.pareto))
    # 提供可再次输入适配器的真实GIS文件；内容仍为模拟数据。
    spec = request["spatial_layers"]
    if spec["kind"] == "raster_grid":
        import rasterio
        from affine import Affine
        fields = list(spec["layers"])
        array = np.asarray([spec["layers"][key] for key in fields], dtype="float64")
        with rasterio.open(args.output / "MOCK_spatial_evaluations.tif", "w", driver="GTiff", height=array.shape[1],
                           width=array.shape[2], count=len(fields), dtype="float64", crs=spec["crs"],
                           transform=Affine(*spec["transform"]), nodata=-9999) as dataset:
            dataset.write(np.where(np.isfinite(array), array, -9999))
            for index, name in enumerate(fields, 1):
                dataset.set_band_description(index, name)
            dataset.update_tags(data_status="MOCK", source="模拟空间评价网格，非工程调查成果")
        features, _ = grid_features(spec)
        (args.output / "MOCK_spatial_evaluations.json").write_text(json.dumps(
            {"kind": "vector", "crs": spec["crs"], "features": features}, ensure_ascii=False, indent=2), "utf-8")
        (args.output / "raster_band_fields.json").write_text(json.dumps(fields), "utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()), "summary": result["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
