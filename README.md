# Face Mask Detection

Ứng dụng nhận diện khẩu trang sử dụng Streamlit.  

Có 2 chế độ:
- **Ảnh**: Upload ảnh hoặc chụp ảnh webcam → dự đoán có đeo/không đeo khẩu trang.
- **Webcam realtime**: Bật webcam để nhận diện liên tục theo thời gian thực.



## Cấu trúc thư mục

```
project/
│── app.py              # code Streamlit chính
│── train.ipynb         # notebook train model
│── readme.md           # mô tả project
│── requirements.txt    # dependencies
│── model.keras         # model đã train sẵn
│
├── dataset_sample/     # dataset mẫu
└── face_detector/      # pretrained face detector
     ├── deploy.prototxt
     └── res10_300x300_ssd_iter_140000.caffemodel
```
### Link tải face detector
- deploy.prototxt → [Tải tại đây](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)  
- res10_300x300_ssd_iter_140000.caffemodel → [Tải tại đây](https://github.com/opencv/opencv_3rdparty/blob/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)

## Cài đặt & Chạy

### 1. Cài dependencies

```bash
pip install -r requirements.txt
```

### 2. Chạy ứng dụng
```bash
streamlit run app.py
```


## Công nghệ sử dụng

### Cho ứng dụng (app.py)
- [Streamlit](https://streamlit.io/) – UI web app
- [TensorFlow/Keras](https://www.tensorflow.org/) – Load & chạy model
- [OpenCV](https://opencv.org/) – Phát hiện khuôn mặt (SSD Caffe model)
- [Pillow](https://python-pillow.org/) – Xử lý ảnh đầu vào
- [NumPy](https://numpy.org/) – Xử lý ma trận ảnh

### Cho training (train.ipynb)
- [TensorFlow/Keras](https://www.tensorflow.org/) – xây dựng & train model
  - MobileNetV2 (pretrained model trên ImageNet)
  - Dense, GlobalAveragePooling2D, Dropout layers
  - Optimizer: Adam
  - Callback: EarlyStopping
- [ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator) – Data Augmentation
- [Matplotlib](https://matplotlib.org/) – Vẽ biểu đồ loss/accuracy
- [scikit-learn](https://scikit-learn.org/stable/) – Đánh giá model (confusion matrix, classification report)


