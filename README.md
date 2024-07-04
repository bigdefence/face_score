# 나의 외모점수는? 🌟

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-FF4B4B)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-EE4C2C)](https://pytorch.org/)

인공지능을 활용하여 당신의 외모 점수를 예측하는 재미있는 웹 애플리케이션입니다!

## 🌟 주요 기능

- 📸 사용자가 업로드한 이미지에서 얼굴 감지
- 🎭 얼굴 메쉬 시각화
- 🧠 딥러닝 모델을 사용한 외모 점수 예측
- 😄 재미있는 결과 메시지 제공

## 🛠️ 설치 방법

1. 이 저장소를 클론합니다:
   ```
   git clone https://github.com/yourusername/appearance-score-app.git
   cd appearance-score-app
   ```

2. 필요한 패키지를 설치합니다:
   ```
   pip install -r requirements.txt
   ```

3. `train.ipynb` 노트북을 실행하여 `facescore.pt` 모델 파일을 생성합니다:
   ```
   jupyter notebook train.ipynb
   ```
   노트북의 지시사항을 따라 모델을 학습하고 저장하세요.

4. 생성된 `facescore.pt` 모델 파일을 프로젝트 디렉토리에 위치시킵니다.

## 🚀 실행 방법

다음 명령어로 Streamlit 앱을 실행합니다:
```
streamlit run app.py
```

## 📊 사용 방법

1. 웹 브라우저에서 앱에 접속합니다.
2. "PNG 또는 JPG 이미지를 업로드하세요." 버튼을 클릭하여 이미지를 업로드합니다.
3. AI가 당신의 외모를 분석하고 점수를 제공할 때까지 기다립니다.
4. 결과와 재미있는 메시지를 확인합니다!

## 🧰 사용된 기술

- [Streamlit](https://streamlit.io/): 웹 애플리케이션 인터페이스
- [OpenCV](https://opencv.org/): 이미지 처리
- [MediaPipe](https://mediapipe.dev/): 얼굴 감지 및 메쉬 생성
- [PyTorch](https://pytorch.org/): 딥러닝 모델 학습 및 실행
- [Albumentations](https://albumentations.ai/): 이미지 전처리

## 🔬 모델 학습

`train.ipynb` 노트북은 외모 점수 예측 모델을 학습하는 데 사용됩니다. 이 노트북에서는:

- 데이터셋 준비
- 모델 아키텍처 정의
- 학습 과정 설정
- 모델 평가
- 최종 모델 저장

의 과정을 거칩니다. 모델 학습에 대한 자세한 내용은 노트북 내의 주석을 참고하세요.

## ⚠️ 주의사항

- 이 앱은 오직 재미를 위한 것이며, 실제 외모를 평가하는 것이 아닙니다.
- 업로드된 이미지는 서버에 저장되지 않습니다.
- 모델의 성능은 학습 데이터에 따라 달라질 수 있습니다.

## 👨‍💻 개발자

이 프로젝트는 [당신의 이름]에 의해 개발되었습니다.

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙏 감사의 글

- [Icons8](https://icons8.com)의 아이콘을 사용하였습니다.
