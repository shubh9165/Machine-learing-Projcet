from setuptools import find_packages,setup
from typing import List
HYPHENDOTE="-e ."

def get_requirments(file_path:str)->List[str]:

    requirments=[]
    with open(file_path) as file_obj:
        requirments=file_obj.readlines()
        requirments=[req.replace("/n","") for req in requirments]

        if HYPHENDOTE in requirments:
            requirments.remove(HYPHENDOTE)
    
    return requirments

setup(
    name='Machine Learning Projcet',
    version='0.0.1',
    author='Shubh',
    author_email='shubhpatel916p@gamil.com',
    packages=find_packages(),
    install_requires=get_requirments('requirment.txt')
)