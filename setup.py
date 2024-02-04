from setuptools import find_packages,setup
from typing import List

const = '-e .'

def get_requirements(file_path:str)->List[str]:

    '''
    To install all the required packages
    '''
    requirements=[]

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        [req.replace("\n","") for req in requirements]

        if const in requirements:
            requirements.remove(const)
    
    return requirements


setup(

name = "Project",
version = "1.0.0",
author = "Pavan Kumar",
author_email = "pavanmalasani@gmail.com",
packages = find_packages(),
install_requires = get_requirements("requirements.txt")

)