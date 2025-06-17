from PIL import Image, ImageDraw

def create_fake_image(path="test.jpg"):
    img = Image.new("RGB", (640, 480), color="gray")  # 创建灰色背景
    draw = ImageDraw.Draw(img)
    draw.rectangle([270, 180, 370, 280], fill="red")  # 画一个红色立方体区域
    img.save(path)
    print(f"Fake test image saved at: {path}")

create_fake_image()
