from setuptools import setup, find_packages

setup(
    name="k_silhouette",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "joblib>=1.4.2"
    ],
    python_requires=">=3.8",
    description="K-Sil Clustering: Silhouette-Guided Instance-Weighted K-Means",
    url="https://github.com/semoglou/ksil",
    author="Angelos Semoglou",
    author_email="a.semoglou@outlook.com",
    license="MIT"
)

