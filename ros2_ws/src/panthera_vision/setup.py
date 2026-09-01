from setuptools import setup

package_name = "panthera_vision"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="aiuser",
    maintainer_email="aiuser@example.com",
    description="ROS2 node for RealSense camera capture and YOLOE detection.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "vision_node = panthera_vision.vision_node:main",
        ],
    },
)
