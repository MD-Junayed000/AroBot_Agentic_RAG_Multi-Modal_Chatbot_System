from setuptools import find_packages, setup

setup(
    name="AroBot",
    version="0.1.0",
    author="Muhammad Junayed, Ridwan Mahmoud",
    author_email="mdjunayed573@gmail.com, u2008039@student.cuet.ac.bd",
    packages=find_packages(),
    install_requires=[],
    description="AroBot: A Conversational Agent on Medical Domain Knowledge Base for Automated Response Generation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown"
)
