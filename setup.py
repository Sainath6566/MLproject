from setuptools import find_packages, setup

def get_requirements(file_path):
    with open(file_path, 'r') as file_obj:
        requirements = [line.strip() for line in file_obj]
    
    requirements = [req for req in requirements if req != '-e .']  # Remove local package reference
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Sainadh',
    author_email='sainathsainath990@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
