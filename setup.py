from setuptools import setup, find_packages

setup(
    name="k_silhouette",
    version="0.1.2",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "joblib>=1.4.2"
    ],
    python_requires=">=3.8",
    description="K-Sil Clustering: A silhouette-guided instance-weighted k-means algorithm that integrates silhouette scores into the clustering process to improve clustering quality.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/semoglou/ksil",
    author="Aggelos Semoglou",
    author_email="a.semoglou@outlook.com",
    license="MIT"
)

