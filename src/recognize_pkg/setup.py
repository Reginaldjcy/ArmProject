from setuptools import find_packages, setup

package_name = 'recognize_pkg'

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
            'keypoints_board_preset = recognize_pkg.keypoints_board_preset:main',
            'keypoints_board = recognize_pkg.keypoints_board:main',
            'keypoints_holistic = recognize_pkg.keypoints_holistic:main',
            'keypoints_hopenet = recognize_pkg.keypoints_hopenet:main',
            'keypoints_pose = recognize_pkg.keypoints_pose:main',
        ],
    },
)
