from setuptools import setup, find_packages

setup(
    name="yolov1-pytorch",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.3.0",
        "torchvision>=0.18.0",
        "pytorch-lightning>=2.2.4",
        "opencv-python-headless",
        "numpy",
        "fastapi",
        "uvicorn",
        "typer",
        "prometheus-client",
        "pydantic>=2.0",
        "optuna",
        "pdoc3",
    ],
    entry_points={
        "console_scripts": [
            "yolov1-train=scripts.train:app",
            "yolov1-detect=scripts.detect:app",
            "yolov1-benchmark=scripts.benchmark:app",
            "yolov1-error-analysis=scripts.error_analysis:app",
            "yolov1-iou-sensitivity=scripts.iou_sensitivity:app",
            "yolov1-hparam-search=scripts.hparam_search:run_search",
            "yolov1-ablation=scripts.ablation:main",
            "yolov1-docs=scripts.generate_docs:generate",
        ]
    },
    python_requires=">=3.11",
)
