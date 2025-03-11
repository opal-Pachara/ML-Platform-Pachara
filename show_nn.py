import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
from torchvision import models


# กำหนดคลาสโมเดลที่คุณต้องการใช้
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        # ใช้โมเดลตัวอย่างจาก torchvision หรือกำหนดโมเดลของคุณเอง
        self.features = models.resnet18(pretrained=True).conv1

    def forward(self, x):
        return self.features(x)

# โหลดโมเดลจากไฟล์ .pth
def load_model(model_path):
    model = CustomModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# แสดงผลโมเดลใน Streamlit
def show_nn():
    model_path = "model/pet_detect_detector.pth"
    
    # โหลดโมเดล
    model = load_model(model_path)

    # ใช้ torchsummary เพื่อแสดงสรุปของโมเดล
    st.write("### Summary of the Model:")
    summary(model, (3, 224, 224))  # กำหนดขนาดของภาพที่โมเดลคาดว่าจะได้รับ

    # แสดงตัวอย่างภาพที่ต้องการทดสอบโมเดล
    image_path = "C:/path/to/image.jpg"
    img = Image.open(image_path)
    st.image(img, caption="Test Image", use_column_width=True)

    # เพิ่มโค้ดสำหรับการทำนายและแสดงผลลัพธ์
    # เชื่อมต่อโค้ดการทำนายตามที่ต้องการ
    st.write("Model loaded and ready for inference!")
