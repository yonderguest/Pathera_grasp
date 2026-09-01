from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def voice_node_parameters(use_voice, voice_prompt_duration):
    """Keep the voice-node enable flag aligned with the grasp brain."""
    return {
        "voice_enabled": use_voice,
        "voice_prompt_duration": voice_prompt_duration,
    }


def generate_launch_description():
    stream_port = LaunchConfiguration("stream_port", default="8080")
    voice_prompt_duration = LaunchConfiguration("voice_prompt_duration", default="3.5")
    use_voice = LaunchConfiguration("use_voice", default="true")
    use_npu = LaunchConfiguration("use_npu", default="false")
    use_graspnet = LaunchConfiguration("use_graspnet", default="false")

    return LaunchDescription(
        [
            DeclareLaunchArgument("stream_port", default_value="8080"),
            DeclareLaunchArgument("voice_prompt_duration", default_value="3.5"),
            DeclareLaunchArgument("use_voice", default_value="true"),
            DeclareLaunchArgument("use_npu", default_value="false"),
            DeclareLaunchArgument("use_graspnet", default_value="false"),
            Node(
                package="panthera_voice",
                executable="voice_node",
                name="panthera_voice",
                parameters=[
                    voice_node_parameters(use_voice, voice_prompt_duration)
                ],
                output="screen",
            ),
            Node(
                package="panthera_vision",
                executable="vision_node",
                name="panthera_vision",
                parameters=[{"use_npu": use_npu}],
                output="screen",
            ),
            Node(
                package="panthera_stream",
                executable="stream_node",
                name="panthera_stream",
                parameters=[{"port": stream_port}],
                output="screen",
            ),
            Node(
                package="panthera_grasp_brain",
                executable="grasp_brain",
                name="panthera_grasp_brain",
                parameters=[
                    {
                        "use_voice": use_voice,
                        "use_npu": use_npu,
                        "use_graspnet": use_graspnet,
                    }
                ],
                output="screen",
            ),
        ]
    )
