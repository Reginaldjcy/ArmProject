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
            'keypoints_face = recognize_pkg.keypoints_face:main',
            'keypoints_hand = recognize_pkg.keypoints_hand:main',
            'keypoints_movenet = recognize_pkg.keypoints_movenet:main',
            'keypoints_pose = recognize_pkg.keypoints_pose:main',
            'keypoints_board = recognize_pkg.keypoints_board:main',
            'keypoints_jeff = recognize_pkg.keypoints_jeff:main',
            'logger = recognize_pkg.logger:main',
            'pkg_area = recognize_pkg.pkg_area:main',
            'pkg_lwh = recognize_pkg.pkg_lwh:main',
            'pkg_open3d = recognize_pkg.pkg_open3d:main',
            'keypoints_hopenet = recognize_pkg.keypoints_hopenet:main',
            'click_point = recognize_pkg.click_point:main',
            'webcam_save = recognize_pkg.webcam_save:main',    
            'icp = recognize_pkg.icp:main',
            'keypoints_undistort = recognize_pkg.keypoints_undistort:main',
            'keypoints_holistic = recognize_pkg.keypoints_holistic:main',
        ],
    },
)
