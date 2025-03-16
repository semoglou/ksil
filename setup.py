from setuptools import setup, find_packages

setup(
    name="ksil",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy==1.26.4",
        "pandas==2.2.2",
        "scikit-learn==1.6.1",
        "scipy==1.14.1",
        "joblib==1.4.2"
    ],
    python_requires=">=3.11",
    description="K-Sil Clustering: Silhouette-Guided Weighted K-Means",
    url="https://github.com/semoglou/ksil",
    author="Angelos Semoglou",
    author_email="a.semoglou@outlook.com",
    license="Apache-2.0"
)

