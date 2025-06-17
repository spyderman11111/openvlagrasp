import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import torch
import numpy as np
import cv2
from PIL import Image as PILImage
from transformers import AutoProcessor, AutoModelForVision2Seq


class OpenVLAGraspNode(Node):
    def __init__(self):
        super().__init__('openvla_grasp_node')

       
        user_input = input("please give a command: ").strip()
        self.prompt = f"In: What action should the robot take to {user_input}?\nOut:"

        self.model_device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.get_logger().info("Loading OpenVLA model...")
        self.processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(self.model_device)
        self.get_logger().info("Model loaded.")

        self.trajectory_pub = self.create_publisher(JointTrajectory, '/scaled_joint_trajectory_controller/joint_trajectory', 10)

        self.home_joint_names = [
            'elbow_joint', 'shoulder_lift_joint', 'shoulder_pan_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        self.home_positions = [0.07505, -1.53058, 1.48675, -0.08071, -1.57239, -0.07874]

        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)

        self.latest_image = None
        self.latest_joints = None
        self.processed = False

        self.init_timer = self.create_timer(1.0, self.send_home_pose)

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
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
        if msg.encoding.lower() == 'bgr8':
            img = img[:, :, ::-1]
        self.latest_image = PILImage.fromarray(img)
        if self.latest_joints is not None and not self.processed:
            self.run_inference()

    def joint_callback(self, msg: JointState):
        if msg.position:
            self.latest_joints = np.array(msg.position[:6], dtype=np.float32)
            if self.latest_image is not None and not self.processed:
                self.run_inference()

    def run_inference(self):
        self.processed = True
        self.get_logger().info("Running inference...")
        self.latest_image = PILImage.open("test.jpg")

        inputs = self.processor(self.prompt, self.latest_image, return_tensors="pt").to(
            self.model_device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )

        
        self.get_logger().info(f"[INPUT] Prompt: {self.prompt}")
        self.get_logger().info(f"[INPUT] Image shape: {np.array(self.latest_image).shape}")
        for k, v in inputs.items():
            self.get_logger().info(f"[INPUT] Tensor: {k}, shape: {tuple(v.shape)}, dtype: {v.dtype}")

        try:
            max_steps = 30
            output = self.model.generate(**inputs, max_new_tokens=max_steps, do_sample=False)

            
            self.get_logger().info(f"[OUTPUT] Raw token ids: {output.tolist()}")

            
            joint_seq = self.model.decode_action(output, unnorm_key="bridge_orig").squeeze(0).detach().cpu().numpy()

            
            self.get_logger().info(f"[OUTPUT] Joint sequence shape: {joint_seq.shape}")
            self.get_logger().info(f"[OUTPUT] First 5 joint positions: {joint_seq[:5, :6].tolist()}")
            self.get_logger().info(f"[OUTPUT] All joint positions (6-DoF only):")
            for i, joint in enumerate(joint_seq[:, :6]):
                self.get_logger().info(f"  Step {i+1:02d}: {joint.tolist()}")

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
        self.get_logger().info(f"Published trajectory with {len(traj.points)} points and step_duration={step_duration:.2f}s.")


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
