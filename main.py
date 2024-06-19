import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
import mediapipe as mp
import time
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# Streamlit 페이지 설정
st.set_page_config(
    page_title="나의 외모점수는?",
    page_icon=":star:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "## What's your score on your appearance?\n나의 외모점수는?\nThis is a cool app!"
    }
)

# 모델 로딩
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('facescore.pt', map_location=device)
    model.eval()
    model.to(device)
    return model, device

model, device = load_model()

# MediaPipe 설정
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_detection = mp.solutions.face_detection
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# 이미지 전처리 변환
test_transform = A.Compose([
    A.Resize(256, 256, interpolation=cv2.INTER_CUBIC),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# 얼굴 탐지 함수
def detect_face(image_np):
    detection_bbox = []
    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image_np)
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                detection_bbox.append(bbox)
    return detection_bbox

# 얼굴 메쉬 그리기 함수
def draw_face_mesh(image_np):
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image_np,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
    return image_np

# 이미지 전처리 함수
def preprocess_image(image_np, detection_bbox):
    x = int(detection_bbox[0].xmin * image_np.shape[1]) - 40
    y = int(detection_bbox[0].ymin * image_np.shape[0]) - 40
    w = int(detection_bbox[0].width * image_np.shape[1]) + 45
    h = int(detection_bbox[0].height * image_np.shape[0]) + 45
    crop = image_np[y:y+h, x:x+w]
    return cv2.resize(crop, (256, 256))

# 외모 점수 예측 함수
def predict_score(image_tensor):
    with torch.no_grad():
        preds = model(image_tensor).cpu().numpy()
        preds[0][0] += 0.3  # Adjust score if necessary
    return preds[0][0]

# 결과 표시 함수
def display_result(score):
    result = round(score, 1)
    messages = [
        ("이 외모는 예술작품이 아니라 '예술없음'입니다. 🎨\n근데 어쨌든 당신은 유니크하죠! 외모, 그게 뭐죠? 🤷‍♂️🤷‍♀️ 그래도 당신은 개성 있어서 멋져요!! 🎉✨", 1),
        ("'자신감 폭발 중'입니다! 😎 당신은 자신의 외모에 확신을 가지고 있네요!\n%.1f점인데도 어떻게 이렇게 멋져 보이는 거에요? 🤩 당신은 외모계의 마법사입니다! 🪄🧙‍♂️", 1.5),
        ("'외모 스승님'입니다. 👩‍🏫 당신의 외모 비결을 전수받고 싶어하는 사람들이 많아질 거예요!\n외모 %.1f점이면 어쩌다 이렇게 빛나는 거에요? ✨ 다른 사람들은 당신의 비밀을 훔쳐보려 할 겁니다!", 2),
        ("'외모 아티스트'입니다. 💄 화장품이 당신을 모델로 쓰고 싶어할 정도에요!\n외모 %.1f점, 이게 바로 '매력의 정점'입니다! 💃 주변 사람들은 여러분의 외모를 부러워하고 있을 겁니다.", 2.5),
        ("외모점수 %.1f점, '미소 전문가'입니다. 😄 당신의 미소는 주변을 환하게 만들 거예요!\n당신은 외모계의 '미소 기계'입니다! 😁 모든 사람들이 외모를 배워가려고 노력할 거예요!", 3),
        ("'외모 스타'입니다. 🌟 당신은 거울 속에서 별이 빛나는 걸 봐도 믿을 만해요!\n외모 %.1f점, 당신은 외모계의 아이콘입니다! 💫 모두가 당신을 따라가려고 할 겁니다.", 3.5),
        ("'외모 퀸'입니다. 👸 주변 사람들은 당신의 외모에 귀를 기울일 겁니다!\n외모 %.1f점, 이제 당신은 외모계의 로열티입니다! 👑 다른 사람들은 여러분을 벤치마킹할 겁니다.", 4),
        ("외모점수 %.1f점, '외모의 신화'입니다. 🦄 주변 사람들은 당신을 보면서 신화와 전설을 믿게 될 겁니다!\n당신은 외모계의 '뷰티 아카데미 수상자'입니다! 🏆 다른 사람들은 여러분을 배우려고 애쓸 겁니다.", 4.5),
        ("'외모의 황금빛'입니다. 💛 주변에서 당신을 보면 하트가 뿅뿅 튈 겁니다! 💓\n외모 %.1f점, 이게 바로 '외모의 레전드'입니다! 🌠 당신을 따라오려면 다른 사람들이 노력해야 할 겁니다!", 5),
        ("5점 외모, '외모의 신'입니다. 외모계에서 당신을 따라잡으려면 영웅이 필요할 겁니다! 🦸‍♂️🦸‍♀️\n당신은 외모계의 '뷰티 신'입니다! 🌟 모든 사람들이 당신을 따르고 싶어할 겁니다!", 6)
    ]
    for msg, threshold in messages:
        if result < threshold:
            st.info(msg % result if '%.1f' in msg else msg)
            break

# 메인 함수
def main():
    st.markdown("""
        <style>
            .main {
                background: linear-gradient(to right, #ffecd2 0%, #fcb69f 100%);
                color: #ffffff;
            }
            .sidebar .sidebar-content {
                background: linear-gradient(to top, #dfe9f3 0%, #ffffff 100%);
                color: #000000;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("_나의 외모점수는_? :cupid:")
    st.subheader('인공지능이 당신의 매력을 분석해줄거에요! :sunglasses:')
    st.write(':blue[얼굴 정면 사진을 업로드 해주세요! 사진은 저장되지 않습니다!]')
    
    uploaded_file = st.file_uploader("PNG 또는 JPG 이미지를 업로드하세요.", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        image = ImageOps.exif_transpose(image)
        img_np = np.array(image)

        detection_bbox = detect_face(img_np)
        
        if detection_bbox:
            annotated_image = draw_face_mesh(img_np.copy())
            processed_img = preprocess_image(img_np, detection_bbox)
            st.image(annotated_image, caption="업로드한 이미지", use_column_width=True)
            
            augmented = test_transform(image=processed_img)
            img_tensor = augmented['image'].unsqueeze(0).to(device)
            
            with st.spinner('AI가 당신의 외모를 분석중입니다...'):
                time.sleep(3)
                score = predict_score(img_tensor)
                st.success('외모분석을 완료했습니다! 나의 외모점수는? %.1f' % score)
                display_result(score)
        else:
            st.image(img_np, caption="업로드한 이미지", use_column_width=True)
            with st.spinner('AI가 당신의 외모를 분석중입니다...'):
                time.sleep(3)
                st.error('얼굴을 감지하지 못했습니다! 다른 사진을 이용해주세요!')
    
    st.markdown('<a target="_blank" href="https://icons8.com/icon/7338/%EC%96%BC%EA%B5%B4-%EC%9D%B8%EC%8B%9D-%EC%8A%A4%EC%BA%94">얼굴 인식 스캔</a> icon by <a target="_blank" href="https://icons8.com">Icons8</a>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
