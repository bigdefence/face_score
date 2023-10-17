import streamlit as st
import cv2
import numpy as np
from PIL import Image,ImageDraw
from keras.models import load_model
import mediapipe as mp
import time
kakao_ad_code = """
 <ins class="kakao_ad_area" style="display:none;"
data-ad-unit = "DAN-8eL7bm4TWXmwWKYS"
data-ad-width = "250"
data-ad-height = "250"></ins>
<script type="text/javascript" src="//t1.daumcdn.net/kas/static/ba.min.js" async></script>
"""

coupang_ad_code="""
<iframe src="https://ads-partners.coupang.com/widgets.html?id=718831&template=carousel&trackingCode=AF3660738&subId=&width=680&height=140&tsource=" width="680" height="140" frameborder="0" scrolling="no" referrerpolicy="unsafe-url"></iframe>
<style>margin: 0 auto;</style>
"""
model=load_model('face.h5')
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_detection = mp.solutions.face_detection
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1,color=(0,255,0))
def main():
	
	st.title("_나의 외모점수는_?:cupid:")
    
    # 파일 업로드 섹션 디자인
	st.subheader('인공지능이 당신의 매력을 분석해줄거에요!:sunglasses:')
	st.write(':blue[얼굴 정면 사진을 업로드 해주세요! 사진은 저장되지 않습니다!]')
    # 파일 업로드 컴포넌트
	uploaded_file = st.file_uploader("PNG 또는 JPG 이미지를 업로드하세요.", type=["png", "jpg", "jpeg"])
	if uploaded_file is not None:
        # 이미지를 넘파이 배열로 변환
		image = Image.open(uploaded_file)
		img_np=np.array(image)
		detection_bbox=[]
		try:
			with mp_face_detection.FaceDetection(
			    model_selection=1, min_detection_confidence=0.5) as face_detection:
			    # 작업 전에 BGR 이미지를 RGB로 변환합니다.
			    results = face_detection.process(img_np)
			    # 이미지를 출력하고 그 위에 얼굴 박스를 그립니다.
			    annotated_image = img_np.copy()
			    for detection in results.detections:
			    	bbox=detection.location_data.relative_bounding_box
			    	detection_bbox.append(bbox)
			with mp_face_mesh.FaceMesh(
			        static_image_mode=True,
			        max_num_faces=1,
			        refine_landmarks=True,
			        min_detection_confidence=0.5) as face_mesh:
			    img = annotated_image
			        # 작업 전에 BGR 이미지를 RGB로 변환합니다.
			    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
			    annotated_image = img.copy()
			    for face_landmarks in results.multi_face_landmarks:
			        mp_drawing.draw_landmarks(
			            image=annotated_image,
			            landmark_list=face_landmarks,
			            connections=mp_face_mesh.FACEMESH_TESSELATION,
			            landmark_drawing_spec=drawing_spec,
			            connection_drawing_spec=mp_drawing_styles
			            .get_default_face_mesh_tesselation_style())
			x=int(detection_bbox[0].xmin*image.width)-35
			y=int(detection_bbox[0].ymin*image.height)-35
			w=int(detection_bbox[0].width*image.width)+40
			h=int(detection_bbox[0].height*image.height)+40
			crop=img_np[y:y+h,x:x+w]
			st.image(annotated_image, caption="업로드한 이미지", use_column_width=True)
			img_resized=cv2.resize(crop,(350,350))
			img_resized=img_resized.astype(np.float32)/255.0
			img_result=[img_resized]
			img_result=np.array(img_result,dtype=np.float32)
			preds=model.predict(img_result)
	        # AI 외모 분석 진행
			with st.spinner('AI가 당신의 외모를 분석중입니다...'):
				time.sleep(3)  # 예시로 3초 동안 로딩 중 표시 (실제 분석으로 대체 필요)
				st.success('외모분석을 완료했습니다!\n나의 외모점수는? %.1f'%preds[0][0])
			result=round(preds[0][0],1)
			if 0<=result<1:
				st.write("이 외모는 예술작품이 아니라 '예술없음'입니다. 🎨\n근데 어쨌든 당신은 유니크하죠! 외모, 그게 뭐죠? 🤷‍♂️🤷‍♀️ 그래도 당신은 개성 있어서 멋져요!! 🎉✨")
			elif 1<=result<1.5:
				st.write("'자신감 폭발 중'입니다! 😎 당신은 자신의 외모에 확신을 가지고 있네요!\n%.1f점인데도 어떻게 이렇게 멋져 보이는 거에요? 🤩 당신은 외모계의 마법사입니다! 🪄🧙‍♂️" % preds[0][0])
			elif 1.5<=result<2:
				st.write("'외모 스승님'입니다. 👩‍🏫 당신의 외모 비결을 전수받고 싶어하는 사람들이 많아질 거예요!\n외모 %.1f점이면 어쩌다 이렇게 빛나는 거에요? ✨ 다른 사람들은 당신의 비밀을 훔쳐보려 할 겁니다!" % preds[0][0])
			elif 2<=result<2.5:
				st.write("'외모 아티스트'입니다. 💄 화장품이 당신을 모델로 쓰고 싶어할 정도에요!\n외모 %.1f점, 이게 바로 '매력의 정점'입니다! 💃 주변 사람들은 여러분의 외모를 부러워하고 있을 겁니다." % preds[0][0])
			elif 2.5<=result<3:
				st.write("외모점수 %.1f점, '미소 전문가'입니다. 😄 당신의 미소는 주변을 환하게 만들 거예요!\n당신은 외모계의 '미소 기계'입니다! 😁 모든 사람들이 외모를 배워가려고 노력할 거예요!" % preds[0][0])
			elif 3<=result<3.5:
				st.write("'외모 스타'입니다. 🌟 당신은 거울 속에서 별이 빛나는 걸 봐도 믿을 만해요!\n외모 %.1f점, 당신은 외모계의 아이콘입니다! 💫 모두가 당신을 따라가려고 할 겁니다." % preds[0][0])
			elif 3.5<=result<4:
				st.write("'외모 퀸'입니다. 👸 주변 사람들은 당신의 외모에 귀를 기울일 겁니다!\n외모 %.1f점, 이제 당신은 외모계의 로열티입니다! 👑 다른 사람들은 여러분을 벤치마킹할 겁니다." % preds[0][0])
			elif 4<=result<4.5:
				st.write("외모점수 %.1f점, '외모의 신화'입니다. 🦄 주변 사람들은 당신을 보면서 신화와 전설을 믿게 될 겁니다!\n당신은 외모계의 '뷰티 아카데미 수상자'입니다! 🏆 다른 사람들은 여러분을 배우려고 애쓸 겁니다." % preds[0][0])
			elif 4.5<=result<5:
				st.write("'외모의 황금빛'입니다. 💛 주변에서 당신을 보면 하트가 뿅뿅 튈 겁니다! 💓\n외모 %.1f점, 이게 바로 '외모의 레전드'입니다! 🌠 당신을 따라오려면 다른 사람들이 노력해야 할 겁니다!" % preds[0][0])
			elif result==5:
				st.write("5점 외모, '외모의 신'입니다. 외모계에서 당신을 따라잡으려면 영웅이 필요할 겁니다! 🦸‍♂️🦸‍♀️\n당신은 외모계의 '뷰티 신'입니다! 🌟 모든 사람들이 당신을 따르고 싶어할 겁니다!")
		except:
			st.subheader('얼굴을 감지하지 못했습니다! 얼굴 정면 사진을 다시 입력해주세요!')
	
	st.components.v1.html(f"{kakao_ad_code}", scrolling=False)
	st.components.v1.html(coupang_ad_code, scrolling=False)
if __name__ == "__main__":
    main()
