from setuptools import setup, find_packages


with open('./requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='DeepLocRNA',
    version='0.0.1',
    author='TerminatorJ',
    author_email='wangjun19950708@gmail.com',
    description='Predicting RNA localization based on RBP binding information',
    license='MIT',
    url='https://github.com/TerminatorJ/DeepLocRNA',
    packages=find_packages(),
    install_requires=requirements,
)
