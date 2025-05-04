from setuptools import find_packages, setup

package_name = 'node_calculation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='reginald',
    maintainer_email='reginaldjcy@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'brd_edge_dist = node_calculation.brd_edge_dist:main',
            'brd_dist_time = node_calculation.brd_dist_time:main',
            'spk_look_brd = node_calculation.spk_look_brd:main',
            'spk_look_brd_hopenet = node_calculation.spk_look_brd_hopenet:main',
            'time_no_brd = node_calculation.time_no_brd:main',
        ],
    },
)
