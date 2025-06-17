import rclpy
from rclpy.node import Node
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

class OpenVLAInferNode(Node):
    def __init__(self):
        super().__init__("openvla_grasp_node")
        self.get_logger().info("Loading model...")

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            attn_implementation=None,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(self.device)

        self.get_logger().info("Model loaded.")
        self.run_inference()

    def run_inference(self):
        image_path = "/mnt/3TB_indego/shuo/py3.10_ros2humble/ros2_ws/src/openvla_grasp/openvla_grasp/test.jpg"
        self.get_logger().info(f"Running inference on image: {image_path}")

        image = Image.open(image_path).convert("RGB")
        prompt = "In: What action should the robot take to pick the red cube?\nOut:"

        inputs = self.processor(prompt, image).to(
            self.device,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )

        with torch.no_grad():
            action = self.model.predict_action(
                **inputs,
                unnorm_key="bridge_orig",
                do_sample=False
            )

        self.get_logger().info(f"Predicted action: {action.tolist()}")
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = OpenVLAInferNode()


if __name__ == "__main__":
    main()
