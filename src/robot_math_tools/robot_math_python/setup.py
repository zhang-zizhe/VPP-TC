from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['pyquaternion', 'modern_robotics'],
    package_dir={'': 'src'}
)

setup(**d)
