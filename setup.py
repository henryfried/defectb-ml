from setuptools import setup, find_packages
import os
import codecs

#here = os.path.abspath(os.path.dirname(__file__))
#with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
#    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'AI for defect tight-binding'
LONG_DESCRIPTION = 'Different machine learning architectures for predicting tight-bidning aprameters for defects from local denstiy of states'

# Setting up
setup(
    name="defectb_ai",
    version=VERSION,
    author="Henry Fried",
    author_email="<henry-fried@hotmail.de>",
    packages=["defectb_ai"],
    package_data={"defectb_ai": ["./data_loader/*.py", "./models/*.py"]},
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    # long_description=long_description,
#    install_requires=['torch', 'pytorch-lightning'],
#    keywords=['pytorch'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
    ]
)
