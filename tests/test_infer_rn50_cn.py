import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "infer_rn50_cn.py"
IMAGE_PATH = REPO_ROOT / "docs" / "CLIP.png"


def test_infer_rn50_cn_help_runs():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--image" in result.stdout
    assert "--text" in result.stdout
    assert "--benchmark-cifar10" in result.stdout
    assert "--num-images" in result.stdout
    assert "--warmup" in result.stdout
    assert "--data-root" in result.stdout


def test_infer_rn50_cn_benchmark_outputs_metrics():
    env = {**dict(), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--benchmark-cifar10",
            "--num-images",
            "1",
            "--warmup",
            "0",
            "--data-root",
            str(REPO_ROOT / "data_test"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0
    assert "benchmark: CIFAR-10 test split" in result.stdout
    assert "metrics:" in result.stdout
    assert "text_count: 10" in result.stdout
    assert "avg_total_seconds_per_image:" in result.stdout
    assert "fps:" in result.stdout


def test_infer_rn50_cn_rejects_invalid_benchmark_counts():
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--benchmark-cifar10",
            "--num-images",
            "0",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--num-images 必须是正整数" in result.stderr
