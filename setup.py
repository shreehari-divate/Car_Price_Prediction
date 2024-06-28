'''
* This file is used to package python libraries and applications.
* Here it is useful for managing dependencies and ensuring consistent environment setup.
* It tells setup tools how to build, install and package the project.
* the setup function is entry point for setuptools. It takes arguments such as:
  -> name
  -> version of project
  -> author
  -> install_requires(this tells setuptools what packages our project requires)
'''



from typing import List
from setuptools import find_packages, setup

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
       * This function reads the requirements from requirement.txt and returns them as list of strings
       * It also removes -e. line from the list. This -e. tells pip to install package in edit mode.
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","")for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements        
setup(
    name='car_price_prediction',
    version='0.0.1',
    author='Shreehari',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)