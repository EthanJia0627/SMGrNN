from setuptools import setup, find_packages

setup(
    name="SMGrNN",  
    version="0.1",  
    packages=find_packages(), 
    install_requires=[ 
        "numpy",
        "torch",
        "networkx",
        "matplotlib",
        "torch-geometric",
    ],
    # entry_points={  # 可选，定义可执行命令行工具
    #     "console_scripts": [
    #         "mypackage-cli=mypackage.module1:main",
    #     ],
    # },
    author="Yiyang Jia",
    author_email="804537158@qq.com",
    description="A package for SMGrNN(Self-Motivated Growing Neural Network)",
    url="https://github.com/EthanJia0627/SMGrNN",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)