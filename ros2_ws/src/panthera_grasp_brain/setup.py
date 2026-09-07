from setuptools import setup

package_name = "panthera_grasp_brain"

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
    description="ROS2 node coordinating Panthera grasping via voice and vision topics.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "grasp_brain = panthera_grasp_brain.grasp_brain_node:main",
        ],
    },
)
