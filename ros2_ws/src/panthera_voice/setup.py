from setuptools import setup

package_name = "panthera_voice"

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
    description="ROS2 node wrapping the offline ASR/TTS voice interface.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "voice_node = panthera_voice.voice_node:main",
        ],
    },
)
