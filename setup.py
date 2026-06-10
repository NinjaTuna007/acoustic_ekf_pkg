from setuptools import find_packages, setup
import os, glob

package_name = 'acoustic_ekf_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # xml launches
        (os.path.join('share', package_name, 'launch'), glob.glob('launch/*.launch')),
        # py launches
        (os.path.join('share', package_name, 'launch'), glob.glob('launch/*.py')),
        # yaml configs
        (os.path.join('share', package_name, 'config'), glob.glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Shekhar Devm Upadhyay',
    maintainer_email='sdup@kth.se',
    description='ROS 2 package for real-time vehicle localization using EKF',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ekf_node = acoustic_ekf_pkg.ekf_node:main',
            'ukf_node = acoustic_ekf_pkg.ukf_node:main',
            'imu_integration_toy_node = acoustic_ekf_pkg.imu_integration_toy_node:main',
            'odom_integration_node = acoustic_ekf_pkg.odom_integration_node:main',
            'trajectory_estimator_node = acoustic_ekf_pkg.trajectory_estimator_node:main',
        ],
    },
)