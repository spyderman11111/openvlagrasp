import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import torch
import numpy as np
from PIL import Image as PILImage
from transformers import AutoProcessor, AutoModelForVision2Seq


class OpenVLAGraspNode(Node):
    def __init__(self):
        super().__init__('openvla_grasp_node')

        self.debug = True  # False 

        user_input = input("Please give a command: ").strip()
        self.prompt = f"In: What action should the robot take to {user_input}?\nOut:"

        self.model_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using device: {self.model_device}")

        self.get_logger().info("Loading OpenVLA model...")
        self.processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

        self.model = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(self.model_device)
        self.get_logger().info("Model loaded.")

        self.trajectory_pub = self.create_publisher(
            JointTrajectory, '/scaled_joint_trajectory_controller/joint_trajectory', 10
        )

        self.home_joint_names = [
            'elbow_joint', 'shoulder_lift_joint', 'shoulder_pan_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        self.home_positions = [0.07505, -1.53058, 1.48675, -0.08071, -1.57239, -0.07874]

        self.latest_image = None
        self.latest_joints = None
        self.processed = False

        self.init_timer = self.create_timer(1.0, self.send_home_pose)

        if not self.debug:
            self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
            self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        else:
            self.get_logger().info("Debug mode enabled. Will run inference with test image.")
            self.run_inference()

    def send_home_pose(self):
        traj = JointTrajectory()
        traj.joint_names = self.home_joint_names
        point = JointTrajectoryPoint()
        point.positions = self.home_positions
        point.velocities = [0.2] * len(self.home_positions)
        point.time_from_start = Duration(sec=3)
        traj.points = [point]
        self.trajectory_pub.publish(traj)
        self.get_logger().info("Sent home position.")
        self.init_timer.cancel()
        self.processed = False

    def image_callback(self, msg: Image):
        try:
            img = np.ndarray(shape=(msg.height, msg.width, 3), dtype=np.uint8, buffer=msg.data)
            if msg.encoding.lower() == 'bgr8':
                img = img[:, :, ::-1]
            self.latest_image = PILImage.fromarray(img)
        except Exception as e:
            self.get_logger().warn(f"[Warning] Image processing failed: {e}")
            return

        if self.latest_joints is not None and not self.processed:
            self.run_inference()

    def joint_callback(self, msg: JointState):
        try:
            if len(msg.position) < 6:
                self.get_logger().warn("Received joint state has fewer than 6 values.")
                return
            self.latest_joints = np.array(msg.position[:6], dtype=np.float32)
        except Exception as e:
            self.get_logger().warn(f"[Warning] Joint state processing failed: {e}")
            return

        if self.latest_image is not None and not self.processed:
            self.run_inference()

    def run_inference(self):
        self.processed = True
        self.get_logger().info("Running inference...")

        if self.debug:
            test_img_path = "/mnt/3TB_indego/shuo/py3.10_ros2humble/ros2_ws/src/openvla_grasp/openvla_grasp/test.jpg"
            if os.path.exists(test_img_path):
                self.latest_image = PILImage.open(test_img_path)
                self.get_logger().info(f"Loaded debug image from {test_img_path}")
            else:
                self.get_logger().error(f"Debug image not found at: {test_img_path}")
                return
        else:
            if self.latest_image is None or self.latest_joints is None:
                self.get_logger().warn("Waiting for both image and joint state to be ready.")
                return

        try:
            # 构建输入（不使用 .to() 直接加在 dict 上）
            inputs = self.processor(
                self.prompt,
                self.latest_image,
                return_tensors="pt"
            )

            # 手动将每个 tensor 移到 device 上
            for k in inputs:
                if isinstance(inputs[k], torch.Tensor):
                    inputs[k] = inputs[k].to(
                        self.model_device,
                        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
                    )

            # 模型推理
            with torch.no_grad():
                action = self.model.predict_action(
                    **inputs,
                    unnorm_key="bridge_orig",
                    do_sample=False
                )

            # 转为 numpy
            if isinstance(action, torch.Tensor):
                joint_seq = action.detach().cpu().numpy()
            else:
                joint_seq = np.array(action, dtype=np.float32)

            # 打印输出确认
            self.get_logger().info(f"[OUTPUT] Trajectory shape: {joint_seq.shape}")
            self.get_logger().info(f"[OUTPUT] First step: {joint_seq[0]}")
            self.get_logger().info(f"[OUTPUT] Last step: {joint_seq[-1]}")

            if joint_seq.shape[1] < 6:
                self.get_logger().error(f"Decoded joint sequence has insufficient DoF: shape={joint_seq.shape}")
                return

            # 只发前6个关节
            self.send_trajectory(joint_seq[:, :6])
        except Exception as e:
            self.get_logger().error(f"[ERROR] Inference failed: {e}")


    def send_trajectory(self, joint_sequence, step_duration=0.5):
        traj = JointTrajectory()
        traj.joint_names = self.home_joint_names
        for i, joint_pos in enumerate(joint_sequence):
            point = JointTrajectoryPoint()
            point.positions = joint_pos.tolist()
            point.time_from_start = Duration(
                sec=int((i + 1) * step_duration),
                nanosec=int(((i + 1) * step_duration % 1) * 1e9)
            )
            traj.points.append(point)

        self.trajectory_pub.publish(traj)
        self.get_logger().info(f"Published trajectory with {len(traj.points)} points.")


def main(args=None):
    rclpy.init(args=args)
    node = OpenVLAGraspNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
