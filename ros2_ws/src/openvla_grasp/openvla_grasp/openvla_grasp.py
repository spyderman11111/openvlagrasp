import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration  # 用于节流时间控制
from sensor_msgs.msg import Image as RosImage, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as RosDuration  # 用于设置 UR5 动作时间
from cv_bridge import CvBridge

import torch
import numpy as np
import cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq


class OpenVLAUR5Controller(Node):
    def __init__(self):
        super().__init__('openvla_ur5_controller_node')
        self.get_logger().info("Loading OpenVLA model...")

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        local_model_path = "/mnt/3TB_indego/shuo/py3.10_ros2humble/openvla-7b"

        self.processor = AutoProcessor.from_pretrained(
            local_model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            local_model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)

        self.prompt = self.get_prompt_once()
        self.get_logger().info(f"Prompt set: {self.prompt}")

        self.bridge = CvBridge()
        self.unnorm_key = "bridge_orig"

        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]

        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            "/scaled_joint_trajectory_controller/joint_trajectory",
            10
        )

        self.image_subscription = self.create_subscription(
            RosImage,
            "/camera/color/image_raw",
            self.image_callback,
            10
        )

        self.joint_state_subscription = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10
        )

        self.current_joint_state = None

        # ==========【节流参数：图像每 N 秒处理一次】=========
        self.process_interval = Duration(seconds=2.0)  # << 修改这里可以改变推理频率
        self.last_processed_time = self.get_clock().now() - self.process_interval
        # ===============================================

        # ==========【机械臂动作耗时设置】=========
        self.motion_duration = 2.0  # << 修改这里可以控制 UR5 每次动作用时（单位：秒）
        # ======================================

        self.image_received = False
        self.create_timer(1.0, self.image_watchdog)

        self.get_logger().info("OpenVLA-UR5 node ready. Executing...")

    def image_watchdog(self):
        if self.prompt and not self.image_received:
            self.get_logger().info("[STATUS] Waiting for image...")

    def get_prompt_once(self):
        user_input = input("Enter a prompt (e.g., 'pick the red cube'):\n")
        return f"In: {user_input.strip()}\nOut:" if user_input.strip() else "In: pick the red cube\nOut:"

    def joint_state_callback(self, msg: JointState):
        joint_info = dict(zip(msg.name, msg.position))
        self.current_joint_state = joint_info

    def image_callback(self, msg: RosImage):
        # ==========【图像节流逻辑：只处理每 N 秒一帧】=========
        now = self.get_clock().now()
        if (now - self.last_processed_time) < self.process_interval:
            return
        self.last_processed_time = now
        # ======================================================

        if not self.image_received:
            self.image_received = True

        self.get_logger().info("[EVENT] Received image, running inference...")

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            pil_image = Image.fromarray(cv_image)
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        inputs = self.processor(self.prompt, pil_image).to(
            self.device,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )

        with torch.no_grad():
            try:
                action = self.model.predict_action(
                    **inputs,
                    unnorm_key=self.unnorm_key,
                    do_sample=False
                )
                action = np.round(action, 4)

            
                self.get_logger().info(f"[PROMPT] {self.prompt}")
                stamp = msg.header.stamp
                self.get_logger().info(f"[TIME] {stamp.sec}.{stamp.nanosec:09d}")

                if self.current_joint_state:
                    joint_str = ", ".join(
                        [f"{k}: {v:.4f}" for k, v in self.current_joint_state.items()
                         if k in self.joint_names]
                    )
                    self.get_logger().info(f"[JOINT] {joint_str}")
                else:
                    self.get_logger().warn("[JOINT] No joint state received yet.")

                self.get_logger().info(f"[ACTION] {action.tolist()}")

                if len(action) >= 6:
                    self.send_joint_command(action[:6])
                else:
                    self.get_logger().warn("Predicted action has insufficient dimensions.")
            except Exception as e:
                self.get_logger().error(f"Inference failed: {e}")

    def send_joint_command(self, joint_positions):
        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = joint_positions.tolist()

        # ==========【设置机械臂动作持续时间】=========
        point.time_from_start = RosDuration(
            sec=int(self.motion_duration),
            nanosec=int((self.motion_duration % 1.0) * 1e9)
        )
        # ===========================================

        msg.points.append(point)
        self.trajectory_pub.publish(msg)
        self.get_logger().info("[CMD] Published joint trajectory command.")

def main(args=None):
    rclpy.init(args=args)
    node = OpenVLAUR5Controller()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down due to keyboard interrupt.")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
