from setuptools import setup, find_packages

setup(
    name="vpptc",
    version="1.0.0",
    description="VPP-TC: Viability-Preserving Planning with Torque Constraints",
    author="VPP-TC Authors",
    license="MIT",
    packages=find_packages(exclude=["scripts", "examples", "output"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pybullet>=3.2.5",
        "cvxpy>=1.4.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "trimesh>=3.21.0",
    ],
)
