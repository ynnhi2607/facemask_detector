import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import keras

# ================== Load Model & Detector ==================
if "model" not in st.session_state:
    st.session_state.model = keras.models.load_model("model.keras")

model = st.session_state.model
CLASSES = ['Mask', 'No Mask']

prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)


# ================== Helper Functions ==================
def detect_and_predict_mask(frame, faceNet, maskNet, conf_threshold=0.5):
    """Phát hiện khuôn mặt và dự đoán đeo/không đeo khẩu trang."""
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces, locs, preds = [], [], []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face) / 255.0

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return locs, preds


def annotate(frame, locs, preds):
    """Vẽ bounding box và nhãn dự đoán lên frame."""
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        text = f"{label}: {max(mask, withoutMask) * 100:.2f}%"

        cv2.putText(frame, text, (startX, max(startY - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    return frame


# ================== Sidebar Menu ==================
st.sidebar.title("Menu")
page = st.sidebar.radio("Chọn trang:", ["Ảnh", "Webcam"])


if "last_page" not in st.session_state:
    st.session_state.last_page = page

if st.session_state.last_page != page:
    try:
        camera.release()
    except:
        pass

    for key in ["webcam_upload", "webcam_realtime"]:
        if key in st.session_state:
            st.session_state[key] = False

    st.session_state.last_page = page
    st.empty()


# ================== Trang 1: Ảnh ==================
if page == "Ảnh":
    st.title("Nhận diện khẩu trang - Test ảnh")
    choice = st.radio("Chọn chế độ:", ["Upload ảnh", "Chụp ảnh webcam"])

    if choice == "Upload ảnh":
        uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            # Đọc ảnh bằng OpenCV
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)

            # Phát hiện khuôn mặt và dự đoán
            locs, preds = detect_and_predict_mask(frame, faceNet, model)

            # Vẽ kết quả
            frame_result = annotate(frame.copy(), locs, preds)

            # Hiển thị kết quả
            st.image(cv2.cvtColor(frame_result, cv2.COLOR_BGR2RGB),
                    caption="Kết quả nhận diện",
                    use_container_width=True)
             # Tạo bảng kết quả
            results = []
            for i, (box, pred) in enumerate(zip(locs, preds)):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                label = "Mask" if mask > withoutMask else "No Mask"
                confidence = max(mask, withoutMask) * 100
                results.append({
                    "Face": i + 1,
                    "Vị trí": f"({startX},{startY})-({endX},{endY})",
                    "Kết quả": label,
                    "Độ tin cậy (%)": f"{confidence:.1f}"
                })

            if results:
                st.write("### Chi tiết kết quả")
                st.table(results)
            else:
                st.warning("Không phát hiện khuôn mặt nào trong ảnh.")

    elif choice == "Chụp ảnh webcam":
        run = st.checkbox("Bật webcam", key="webcam_upload")
        FRAME_WINDOW = st.image([])

        if run:
            camera = cv2.VideoCapture(0)
            snapshot_btn = st.button("📸 Chụp ảnh")

            while run and camera.isOpened():
                ret, frame = camera.read()
                if not ret:
                    st.write("Không mở được webcam")
                    break

                frame = cv2.flip(frame, 1)
                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                   caption="Preview", use_container_width=True)

                if snapshot_btn:
                    locs, preds = detect_and_predict_mask(frame, faceNet, model)
                    frame_result = annotate(frame.copy(), locs, preds)
                    st.image(cv2.cvtColor(frame_result, cv2.COLOR_BGR2RGB),
                             caption="Kết quả nhận diện", use_container_width=True)
                    camera.release()
                    break


# ================== Trang 2: Webcam realtime ==================
elif page == "Webcam":
    st.title("Nhận diện khẩu trang - Webcam realtime")
    run = st.checkbox("Bật webcam", key="webcam_realtime")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.write("Không mở được webcam")
            break

        frame = cv2.flip(frame, 1)
        locs, preds = detect_and_predict_mask(frame, faceNet, model)
        frame = annotate(frame, locs, preds)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    camera.release()
