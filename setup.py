from setuptools import setup, find_packages

setup(
    name='capravens',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    license=open('LICENSE').read(),
    zip_safe=False,
    description="CapRavens: Long-Horizon Language-Conditioned Robot Arm Manipulation with code Generation.",
    author='Haihui Ye',
    #  install_requires=[line for line in open('requirements.txt').readlines() if "@" not in line],
    keywords=['Large Language Models', 'Simulation', 'Vision Language Grounding', 'Robotics',
              'Manipulation'],
)
