from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()
   
setup(
   name='kinjax',
   version='0.0.5', 
   description='This module calculates FK and Jacobian, accelerated by Jax to utilize JIT and GPU parallelization',
   long_description    = long_description,
   long_description_content_type='text/markdown',
   author='Kanghyun Kim',
   author_email='kh11kim@kaist.ac.kr',
   packages=find_packages(),  #same as name
   install_requires=[
      "jax",
      "sympy",
      "scipy",
   ]
)

# python setup.py sdist bdist_wheel
# twine upload dist/*
