"""
run_all.py - 一键运行所有阶段的所有案例
=================================================

用法:
    python run_all.py              # 运行所有已完成的案例
    python run_all.py stage1       # 只运行第 1 阶段
    python run_all.py stage1 3     # 只运行第 1 阶段第 3 个案例
"""

import subprocess
import sys
import os

# 强制 stdout 使用 UTF-8，避免 Windows GBK 控制台编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 项目根目录
ROOT = os.path.dirname(os.path.abspath(__file__))

# 各阶段的案例配置
STAGES = {
    "stage1": {
        "name": "NumPy 科学计算工具链",
        "dir": "stage1_numpy",
        "cases": [
            "01_array_basics.py",
            "02_vectorized_ops.py",
            "03_broadcasting.py",
            "04_boolean_indexing.py",
            "05_matrix_ops.py",
            "06_reshape_slice.py",
            "07_random_stats.py",
        ],
    },
    "stage2": {
        "name": "数学基础",
        "dir": "stage2_math",
        "cases": [],
    },
    "stage3": {
        "name": "机器学习核心原理",
        "dir": "stage3_ml",
        "cases": [],
    },
    "stage4": {
        "name": "深度学习原理",
        "dir": "stage4_deep_learning",
        "cases": [],
    },
    "stage5": {
        "name": "Transformer 与大语言模型",
        "dir": "stage5_transformer",
        "cases": [],
    },
}


def run_case(stage_key, case_file):
    """运行单个案例文件"""
    stage = STAGES[stage_key]
    case_path = os.path.join(ROOT, stage["dir"], case_file)

    if not os.path.exists(case_path):
        print(f"  [SKIP] {case_file} (文件不存在)")
        return False

    print(f"\n{'-' * 60}")
    print(f"  [{stage_key}] {case_file}")
    print(f"{'-' * 60}")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [sys.executable, case_path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=ROOT,
        env=env,
    )

    # 打印输出
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(f"  [STDERR] {result.stderr}")

    if result.returncode != 0:
        print(f"  [FAIL] 退出码 {result.returncode}")
        return False

    return True


def run_stage(stage_key):
    """运行一个阶段的所有案例"""
    stage = STAGES[stage_key]
    print(f"\n{'=' * 60}")
    print(f"  阶段: {stage['name']} ({stage_key})")
    print(f"{'=' * 60}")

    if not stage["cases"]:
        print("  (本阶段尚未开发案例)")
        return

    passed = 0
    total = len(stage["cases"])

    for case_file in stage["cases"]:
        if run_case(stage_key, case_file):
            passed += 1

    print(f"\n  {stage_key} 结果: {passed}/{total} 通过")
    return passed == total


def main():
    args = sys.argv[1:]

    if not args:
        # 运行所有阶段
        print("\n" + "=" * 60)
        print("  Python AI Learning - 一键运行所有案例")
        print("=" * 60)
        all_passed = True
        for stage_key in STAGES:
            if not run_stage(stage_key):
                all_passed = False
        print("\n" + "=" * 60)
        if all_passed:
            print("  ALL PASSED")
        else:
            print("  SOME CASES FAILED")
        print("=" * 60)
    elif len(args) == 1:
        # 运行指定阶段
        stage_key = args[0]
        if stage_key in STAGES:
            run_stage(stage_key)
        else:
            print(f"未知阶段: {stage_key}")
            print(f"可选: {', '.join(STAGES.keys())}")
    elif len(args) == 2:
        # 运行指定阶段的指定案例
        stage_key = args[0]
        case_num = int(args[1]) - 1
        if stage_key in STAGES and 0 <= case_num < len(STAGES[stage_key]["cases"]):
            run_case(stage_key, STAGES[stage_key]["cases"][case_num])
        else:
            print(f"无效参数: {stage_key} {args[1]}")


if __name__ == "__main__":
    main()
