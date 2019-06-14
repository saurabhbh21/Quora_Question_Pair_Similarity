from setuptools import setup

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()


setup(
   name='QPSimilarity',
   version='1.0',
   description='Identifying whether two question are same in context',
   license="",
   author='Saurabh Bhagvatula',
   author_email='saurabhbh21@gmail.com',
   packages=['QPSimilarity'],  #same as name
   install_requires=[required], #external packages as dependencies from requirements.txt
   scripts=[
            'scripts/download.sh',
            
           ]
)