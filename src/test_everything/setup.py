from setuptools import find_packages, setup

package_name = 'test_everything'

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
            'keypoints_steady = test_everything.keypoints_steady:main',
            'show_boardmulti = test_everything.show_boardmulti:main',
            'show_hopenet = test_everything.show_hopenet:main',
            'show_point = test_everything.show_point:main',
            'show_point2 = test_everything.show_point2:main',
            'show_pose_brd = test_everything.show_pose_brd:main',
            'show_preset_brd = test_everything.show_preset_brd:main',
            'show_jeff_brd = test_everything.show_jeff_brd:main'

        ],
    },
)
