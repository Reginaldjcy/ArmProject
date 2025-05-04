from setuptools import find_packages, setup

package_name = 'shooting_style'

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
            'flw_spk = shooting_style.flw_spk:main',
            'spk_brd = shooting_style.spk_brd:main'
        ],
    },
)
