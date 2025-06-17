from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='openvla_grasp',
            executable='openvla_grasp',
            name='openvla_grasp_node',
            output='screen',
        )
    ])
