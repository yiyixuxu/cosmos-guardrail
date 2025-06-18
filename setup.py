from setuptools import setup, find_packages

setup(
    name="cosmos-guardrail",
    version="0.1.0",
    description="Guardrail system for content safety in text and video, part of NVIDIA Cosmos project",
    long_description="""
    This package contains content safety guardrails from the NVIDIA Cosmos project.
    It provides tools for checking text and video content safety.
    """,
    author="NVIDIA",
    author_email="cosmos-license@nvidia.com",
    url="https://github.com/nvidia-cosmos/cosmos-predict1",
    license="Apache-2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.6.0",
        "torchvision>=0.21.0",
        "numpy>=1.26.4",
        "transformers>=4.49.0",
        "Pillow>=11.1.0",
        "huggingface-hub>=0.29.2",
        "opencv-python>=4.10.0.84",
        "better-profanity>=0.7.0",
        "nltk>=3.9.1",
        "peft>=0.14.0",
        "retinaface-py>=0.0.2",
        "safetensors>=0.5.3",
        "sentencepiece>=0.2.0",
        "protobuf",
        "attrs>=25.1.0",
        "tqdm>=4.66.5",
        "scikit-image>=0.25.2",
        "requests>=2.28.0",  # For URL downloading in load_video
        "imageio>=2.37.0",
        "imageio-ffmpeg>=0.4.5",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    project_urls={
        "Source": "https://github.com/nvidia-cosmos/cosmos-predict1",
        "Research": "https://research.nvidia.com/labs/dir/cosmos-predict1",
    },
) 
