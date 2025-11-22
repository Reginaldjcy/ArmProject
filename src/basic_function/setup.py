from setuptools import find_packages, setup

package_name = 'basic_function'

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
            'chest_pose = basic_function.chest_pose:main',
            'face_to_arm = basic_function.face_to_arm:main',
            'gesture_pose = basic_function.gesture_pose:main',
            'reid = basic_function.reid:main',
        ],
    },
)
