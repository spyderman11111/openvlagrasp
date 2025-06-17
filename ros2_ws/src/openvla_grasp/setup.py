from setuptools import setup

package_name = 'openvla_grasp'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         [f'resource/{package_name}']),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/openvla_grasp.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your@email.com',
    description='OpenVLA-based grasp node for UR5',
    license='MIT',
    entry_points={
        'console_scripts': [
            'openvla_grasp = openvla_grasp.openvla_grasp:main',
        ],
    },
)
