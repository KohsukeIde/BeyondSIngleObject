from setuptools import find_packages, setup


setup(
    name="beyond-single-object",
    version="0.1.0",
    description="Multi-object 3D relation learning with PointLLM-based 3D-LLMs.",
    packages=find_packages(include=["pointllm", "pointllm.*"]),
    python_requires=">=3.9",
)
