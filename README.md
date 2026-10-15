# 웹툰 → 실사화: 20컷 × 4개 파이프라인 A/B 테스트 실행 계획

**작성 기준:** 2026-10-15 · **환경:** RTX 5070 Ti 16GB / Windows + ComfyUI · **목표:** 웹툰의 행동·구도는 유지하고, 실제 사진 같은 인물 표현과 여러 컷에 걸친 얼굴 동일성을 확보한다.

> **문서 범위:** 실행 설계·ComfyUI 노드 맵·권장 초기 파라미터·데이터셋/기록 양식·자동 QA 평가 기준. 이 문서에 있는 모델과 공개 워크플로우의 존재는 출처를 확인했으나, 네 가지 조합의 통합 구동 및 5070 Ti 성능은 직접 실측하지 않았다. 특히 B/C/D의 출력 품질 순위는 테스트 전에는 확정할 수 없다.

## 0. 가장 중요한 전제와 의사결정

- **A:** 기존 **Depth → 현재 오픈웨이트 I2I 생성기 → Face Swap**. 기준선이므로 모델/노드/강도는 현재 운영값을 그대로 동결. 변경 시 기준선이 달라지므로 `manifest.csv`에 기록한다.
- **B:** **Krea 2 Turbo + Anything2RealCharacters V3**. 원본 웹툰의 **스타일 변환** 중심. 추가 인물 사진을 사용하는 Dual-Image는 실험적이다. 본 실험에서는 레퍼런스 사진을 제공하는 시도와 불가능할 경우의 폴백을 명확히 기록한다.
- **C:** **Krea 2 Turbo + Krea Pose Control LoRA → BFS Krea 2 Head Swap V1.1**. 포즈 지도 기반으로 실사 인물을 만든 뒤 동일 Krea 계열의 전용 BFS로 얼굴/머리를 교정한다. **Qwen용 BFS를 여기 연결하지 않는다.**
- **D:** **Qwen-Image-2.1 + Fisher Pose/VNCCS PoseStudio → BFS Qwen 2.1 Head V1.1**. Fisher 3D 마네킹+인물 사진을 이용한 생성 다음 Qwen 전용 BFS로 얼굴을 교정한다. **Krea용 BFS를 여기 연결하지 않는다.**
- **결과 수:** 20컷 × 4방식 × 3시드 = **240개 최종 후보 이미지**. 각 후보의 중간 결과를 별도로 저장하고, 베이스 생성·BFS·리파인 시간이 각각 얼마인지 기록한다.
- **실험 설계상 주의:** 네 조합은 모델과 전처리/후처리가 함께 달라지는 **파이프라인 비교**이며, 특정 구성요소 하나의 순수 인과효과 비교가 아니다. 중요한 차이가 발견되면 후속 소거 실험(ablation)을 한다.

## 1. 사전 점검 / Go-No-Go 게이트

1. 대상 웹툰 컷의 사용·변환 권한과 인물 사진의 초상/생성 동의를 확인한다. 실존 인물을 사용하는 경우 별도 권한 기록 필수.
2. Krea 2 Turbo 원본은 **게이트 모델**이다. Hugging Face 로그인 후 사용 조건 동의와 다운로드 권한을 먼저 확보한다. Comfy-Org 재패키지 파일도 원 모델 라이선스를 그대로 따른다.
3. Qwen-Image-2.1은 **Qwen Research License(비상업 연구용)**. 공개 또는 상용 웹서비스에는 별도 상업 허가가 필요하다. LoRA 자체의 MIT/Apache 표기가 베이스 모델 사용권을 대체하지 않는다.
4. 같은 20컷, 동일 목표 캐릭터 레퍼런스, 동일 캔버스 비율, 동일 평가 프로토콜을 전 조합에 적용한다. 모델마다 동일한 숫자 시드를 사용하되, **시드 번호가 같아도 모델 간 동일한 난수 조건은 아니다**.
5. **스모크 테스트:** 단일 인물 1컷, 측면 1컷, 전신 1컷을 각 조합에 1시드씩 실행. 링크 누락, 노드 불일치, OOM, 명백한 구조 이탈을 확인하고 본실험 전 설정 고정.
6. 16GB에서 Krea+BFS 또는 Qwen+Fisher를 **한 번에 GPU에 올리지 않는다.** Stage별 실행·모델 언로드/캐시 정리·CPU 오프로딩으로 처리하고 OOM이면 해상도부터 낮춘다.

## 2. 실행환경 구축

### 2.1 ComfyUI / 공통 노드

- 설치/업데이트: [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 최신 호환 릴리스. 기존 실행 환경을 복제하고 패키지 버전 및 Git 커밋을 기록할 것.
- 포즈 추출: [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) — **DWPose Preprocessor**. DWPose가 웹툰에 실패하면 수동 교정 또는 포즈 에디터에서 재지정.
- 이미지 입출력: ComfyUI 기본 `LoadImage`, `SaveImage`, `ImageScale`, `ImageScaleToTotalPixels`, `VAELoader`, `UNETLoader`, `LoraLoader`, `KSampler`/`SamplerCustomAdvanced` 등 **불러온 공식 JSON의 실제 노드/이름을 기준으로** 연결.
- 복잡한 포즈·원근: [Fisher Pose](https://github.com/Work-Fisher/ComfyUI-Fisher-Pose)에서 사람·3D 마네킹 포즈와 카메라 편집.
- Krea 참조 이미지 조건: [ComfyUI-Krea2-Ostris-Edit](https://github.com/ostris/ComfyUI-Krea2-Ostris-Edit); `Text Encode Krea 2 Ostris Edit`(표시 명칭은 버전에 따라 변할 수 있음), 제공 예시 워크플로우 우선.
- **주의:** `Pose ControlNet LoRA`는 일반적인 `ControlNetLoader`와 동일한 형식이라고 가정하면 안 된다. **제작자 JSON에서는 Krea2 Ostris Edit 인코더의 image1에 스켈레톤 맵을 넣고 Krea 2 LoRA로 로드**하도록 명시한다.

### 2.2 설치 명령 예시 (Windows PowerShell, ComfyUI 폴더 기준)

```powershell
cd .\ComfyUI\custom_nodes
git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
git clone https://github.com/ostris/ComfyUI-Krea2-Ostris-Edit.git
git clone https://github.com/Work-Fisher/ComfyUI-Fisher-Pose.git
# 기존 설치물이 있다면 clone하지 말고 해당 저장소의 상태/커밋만 기록
```

모델 파일은 아래 표의 경로로 설치. **Krea·Qwen 체크포인트는 동일 모델이 아니므로 각자의 text encoder/VAE/로라 조합을 지켜야 한다.**

## 3. 다운로드 및 워크플로우 바로가기 (확인된 원문)

| 구분 | 모델/워크플로우 | 저장 위치 / 비고 |
|---|---|---|
| Krea base | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) · [원본 krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) | `models/diffusion_models/`에 `krea2_turbo_fp8_scaled.safetensors`(또는 호환 저정밀본); `models/text_encoders/` `qwen3vl_4b_fp8_scaled.safetensors`; `models/vae/` `qwen_image_vae.safetensors`. Comfy-Org README 참조. |
| B 스타일 | [Krea2 Anything2RealCharacters V3](https://huggingface.co/WarmBloodAban/Krea2_Anything2RealCharacters/tree/main) | `Krea2_Anything2RealCharacters-V3.safetensors` → `models/loras/`; 모델카드 V3 설명 및 저장소 제공 워크플로우 확인. |
| C 포즈 | [thedeoxen Krea-2 Pose Control](https://huggingface.co/thedeoxen/Krea-2-pose-controlnet) | `krea2_turbo_openpose_controlnet.safetensors` → `models/loras/`; [공식 JSON](https://huggingface.co/thedeoxen/Krea-2-pose-controlnet/resolve/main/krea2_controlnet_pose.json). |
| C BFS head | [BFS Krea 2 안내](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/blob/main/docs/krea-2.md) · [BFS 파일 목록](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/tree/main) | `bfs_head_swap_v1.1_krea2.safetensors` → `models/loras/`; `workflows/Head Swap Krea 2 - V1 Simple Workflow.json`. |
| C BFS body (선택) | [BFS Krea 2 안내](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/blob/main/docs/krea-2.md) | `bfs_body_swap_v1_krea2.safetensors` → `models/loras/`; **실험적, 포즈 보존 불확실**. 본 실험 기본 경로에는 미포함. |
| Qwen base | [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) · [Qwen 원본](https://huggingface.co/Qwen/Qwen-Image-2.1) | `models/diffusion_models/` 저정밀 체크포인트, `models/text_encoders/` `qwen3vl_8b_int8_convrot.safetensors` 또는 호환 w4a8, `models/vae/` `qwen_image_2.1_vae_bf16.safetensors`. 제공 README의 정확한 이름 확인. |
| D 포즈 | [Fisher Pose](https://github.com/Work-Fisher/ComfyUI-Fisher-Pose) · [VNCCS PoseStudio LoRA](https://huggingface.co/MIUProject/VNCCS_PoseStudio_QI2.1) | `models/loras/` VNCCS QI2.1 PoseStudio 가중치; Fisher 저장소 최신 공식 워크플로우. |
| D BFS head | [BFS Qwen 2.1 문서](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/blob/main/docs/qwen-image-2.1.md) | `bfs_head_v1.1_qwen_2.1.safetensors` → `models/loras/`; [공식 JSON](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/resolve/main/workflows/Head%20Swap%20V1%20Qwen%202.1%20Workflow.json). |
| D BFS body (선택) | [BFS Qwen 2.1 Body 가이드](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/blob/main/docs/qwen-image-2.1-body.md) | `bfs_body_swap_v1.0_qwen_2.1.safetensors`; 본 비교에서는 Head만 기본 실행. |

**메모:** `BFS` 저장소에는 여러 베이스 모델용 파일이 섞여 있다. 특정 LoRA를 로드할 때 항상 파일명의 `krea2` 또는 `qwen_2.1`을 확인한다.

## 4. 데이터셋: 20컷 균형 설계

| 그룹 | 컷 수 | 포함 조건 | 반드시 점검할 실패 |
|---|---:|---|---|
| G1 얼굴 클로즈업 | 4 | 정면/측면, 열린 입, 서로 다른 표정 | identity, 표정, 피부 질감 |
| G2 상반신/대화 | 4 | 손 제스처, 앉은 자세, 비스듬한 카메라 | 관절·의상·표정 |
| G3 전신/동작 | 4 | 걷기/달리기/굽힘/물건 잡기 | 전체 포즈·손/발 |
| G4 복합 구도 | 4 | 2인 또는 부분 가림, 교차하는 팔/다리 | 인물 혼동, 가림, 인원 수 |
| G5 특수 시점 | 4 | 하이앵글/로우앵글, 과장된 원근·컷 프레이밍 | 구도·시점·해부학 |

실험용 인물은 적어도 동일 캐릭터 **정면 + 좌/우 45도 + 측면 + 상반신/전신** 레퍼런스 사진을 준비한다. 가능하면 각 컷에 동일 인물 1명의 ID를 명확히 지정. 두 인물 등장 컷은 Person-1 / Person-2 인식·크롭·QA를 분리한다. 실사 레퍼런스에 실존 인물을 사용할 경우 사전 동의와 결과 관리 정책을 갖춘다.

**폴더 규격** (ZIP 내 템플릿 파일과 동일)

```text
webtoon_ab_test/
  inputs/comic/C001.png ... C020.png
  inputs/identities/P01/front.png, left45.png, right45.png, side.png, full.png
  annotations/pose/C001.json ... C020.json
  annotations/crops/C001.json ... C020.json
  intermediate/A|B|C|D/C001/seed_001/{stage_01.png,stage_02.png}
  outputs/A|B|C|D/C001/seed_001.png
  logs/runs.csv
  logs/qa_scores.csv
  logs/qa_reviews.csv
  results/leaderboard.csv
```

**공통 변수:** 컷 ID, 인물 ID, 목표 서술 프롬프트, 참조 사진 순서, 생성 해상도/비율, 시드 3개(예 `101`, `202`, `303`), 원본 컷/포즈 파일 해시, 모델 버전/가중치 해시, 워크플로우 JSON 해시를 기록한다.

## 5. A/B/C/D ComfyUI 그래프와 초기 파라미터

### A. 기존 Depth + Face Swap (baseline)

```text
LoadImage(comic) → DepthPreprocessor(현행 노드/모델) → 현행 I2I 조건 연결
→ 현행 생성기(기존 sampler/steps/CFG/denoise 유지) → 현행 Face Swap
→ SaveImage(stage_02_final)
```

- **초기 파라미터:** **현행 설정 그대로 유지**. 현재 값이 제공되지 않았으므로 임의의 수치를 채우지 않는다.
- 기록: 사용 모델/Depth 노드/denoise/Control 강도/Face Swap 모델/검출기/마스크·블렌딩/얼굴 유사도 설정/생성 크기.
- 비교 시 새로운 품질 향상 노드를 A에만 추가하지 말 것.

### B. Krea 2 + Anything2RealCharacters V3

```text
LoadImage(comic) → Krea 2 이미지 편집/참조 조건 노드(저장소 제공 JSON 기준)
→ Krea 2 Turbo FP8 + LoraLoader(Anything2Real V3)
→ Sampler → VAEDecode → SaveImage(stage_01, final)
[실험적] Identity reference를 2번째 이미지 슬롯으로 전달 → 결과와 미지원 여부 기록
```

- **검증된 개발자 권장값:** Steps `8`, CFG `1.0`, 주 트리거 `transform the image to realistic photograph`.
- **VRAM 우선 PoC값:** 768×1024(세로) 또는 1024×768(가로), Euler/Simple 시작. 게시된 2048×2048 권장은 16GB를 고려해 초기 테스트에서 사용하지 않는다.
- **V3 LoRA 강도:** `0.6 / 0.8 / 1.0`으로 **스모크 컷에서만 탐색 후 고정** (공식 최적값 아님).
- **Dual-Image:** 소스 컷=Figure 1, 동일 인물=Figure 2; 예시 문구 `with the character looking exactly like the one in Figure 2.`. **실험적 기능이므로 UI/노드가 해당 이미지 슬롯을 실제 지원하는지 확인**한다. 참조 입력 미지원 시 B는 스타일 전용 arm으로 표기하여 identity 점수 해석 시 차이를 명시한다.
- **실패 관찰:** 원본 셀 셰이딩 잔존, 캐릭터 얼굴 재해석, 피부 플라스틱화.

### C. Krea 2 + Pose Control + BFS (Krea 전용)

```text
LoadImage(comic) → DWPosePreprocessor → [수동/에디터 포즈 수정]
→ LoadImage(pose_black_bg) → Text Encode Krea 2 Ostris Edit(image1 = pose map)
→ Krea2 Turbo FP8 + LoraLoader(krea2_turbo_openpose_controlnet)
→ KSampler/VAEDecode → SaveImage(C_stage_pose)
→ [모델 언로드/필요 시 재시작]
→ LoadImage(C_stage_pose = image1) + LoadImage(identity_ref = image2)
→ BFS Krea Head Swap V1.1 공식 JSON + 해당 Krea LoRA
→ SaveImage(C_final)
```

- **Pose 권장값(제작자):** LoRA `0.8~1.0`, steps `약 10`, CFG `약 1.0`, sampler `euler`, scheduler `simple`.
- **PoC 기본값:** Pose LoRA `0.9`, steps `10`, CFG `1.0`, 768×1024/1024×768.
- **BFS Head:** 공식 Krea 헤드 스왑 JSON부터 사용. `image1 = C_stage_pose`, `image2 = 얼굴 레퍼런스`; LoRA 강도는 스모크에서 `0.75 / 1.0` 실험 후 고정(문서의 Head 강도 확정값은 별도 명시되지 않음).
- **중요:** C에서는 `bfs_head_swap_v1.1_krea2.safetensors`만 로드. Qwen `bfs_head_v1.1_qwen_2.1.safetensors`를 연결하면 안 됨.
- **실패 관찰:** 무너진 팔·손, 포즈 관절 좌표 드리프트, swap 후 머리 크기/피부/광원 불일치.

### D. Qwen-Image-2.1 + Fisher Pose + BFS (Qwen 전용)

```text
LoadImage(comic) → DWPose/포즈 좌표 참조 → Fisher 3D 포즈 에디터
→ 사람 사진 + 3D 마네킹 조건(Fisher 공식 최신 workflow)
→ Qwen-Image-2.1 + VNCCS PoseStudio QI2.1 LoRA
→ TextEncodeQwenImage21(또는 Fisher Qwen2.1 인코더)
→ Sampler → VAEDecode → SaveImage(D_stage_pose)
→ [모델 언로드/필요 시 재시작]
→ BFS Qwen 2.1 Head V1.1 공식 JSON
→ image1 = D_stage_pose, image2 = identity_ref → SaveImage(D_final)
```

- **Fisher 3D:** 레퍼런스 사진과 마네킹의 입력 순서는 **공식 Fisher workflow의 인코더 기준**으로 따라야 한다. VNCCS 방식의 기본 구조는 `image1=마네킹`, `image2=인물 사진`, `Draw character from image2`로 안내된다. 최근 Fisher의 결합 인코더는 인물 레퍼런스가 먼저이고 마네킹을 맨 뒤에 붙이므로 **서로의 슬롯 순서를 혼용 금지**.
- **VNCCS PoseStudio LoRA:** 저장소 안내를 따라 초기 `1.0`으로 시작. Qwen 2.1 초기 steps `25~35`, CFG `1.0`, 768×1024.
- **BFS Qwen head:** `bfs_head_v1.1_qwen_2.1.safetensors`, 초기 강도 `1.0`(개발자 권장), 공식 JSON 사용. 필요 시 0.8과 비교하되 스모크 이후 고정.
- **BFS 참고:** Qwen Body Swap은 별도 LoRA(Head와 다름). 기본 D arm에 포함하지 않는다.
- **실패 관찰:** 마네킹 포즈와 최종 관절 편차, 가림 처리, BFS 후 얼굴 크기·피부 질감, VRAM/CPU 오프로딩 지연.

## 6. VRAM 16GB 운영 가이드

- Krea 기본은 Comfy-Org 배포의 `krea2_turbo_fp8_scaled` 또는 호환 INT8로 시작. Krea 모델이 이미지 입력을 받는 편집 노드는 공식 예시 JSON을 먼저 불러와 검증.
- Qwen 기본은 Comfy-Org 배포의 `qwen_image_2.1_*_int8_convrot` 등 모델과 Qwen3-VL 8B INT8/W4A8 텍스트 인코더를 조합. 정확한 파일명은 해당 버전 README와 디렉터리에서 재검증.
- **Krea ↔ Qwen 전환 시 ComfyUI를 재시작하는 것이 가장 재현성이 높다.** 모델 오프로딩·캐시가 GPU에 남아 생기는 OOM을 줄이기 위해 각 arm을 묶어 일괄 실행.
- 첫 세팅 `--lowvram` 또는 UI 메모리 최적화, 필요 시 CPU 오프로딩. 실행 모델이 느릴 수 있으므로 **warm start 시간/steady-state 시간/피크 VRAM을 따로** 기록.
- 초기 이미지 픽셀 예산 약 0.8MP(예 768×1024). OOM 시 먼저 640×896으로 감소 → lowvram/오프로딩 → GPU 재시작 → 저정밀 weights 검토. 품질 최종 평가에서는 해상도를 가능한 한 동일하게 고정.
- **피해야 할 조합:** 16GB에 Qwen 전체 BF16 + 인코더 BF16 + 추가 LoRA·BFS를 동시에 상주시키는 구성. 각 stage를 개별 실행 후 이미지 파일로 넘길 것.
- 크기/메모리/시간은 개인 환경·Comfy 버전·커스텀 노드에 따라 달라지므로 특정 초당 속도를 보장하지 않는다.

## 7. 실험 실행 절차 및 스코어링

**단계 1 · 준비 (2~3개의 대표 컷):** 매 arm 공식 workflow JSON 단독 실행을 확인한 후 연결 그래프를 수정한다. 각 단계에서 이미지 저장·로라 로드·스왑 입력 순서 확인.

**단계 2 · 스모크 (3컷 × 4 arm × 1 seed = 12개):** Krea B LoRA 강도, C 포즈 강도/Head, D Fisher/BFS 설정을 하나씩 탐색. 스모크 데이터는 20컷 평가와 분리하거나 하이퍼파라미터 선택의 오염을 메타데이터에 명시.

**단계 3 · 본실험 (20컷 × 4 arm × 3 seed = 240개):** 컷별 seed `101/202/303` 동결. 한 arm씩 20컷을 실행해 메모리 전환 비용을 줄인다. 중간/최종 PNG·workflow JSON·prompt·seed·소요시간·피크VRAM을 저장.

**단계 4 · 자동 QA:** 포즈/구도/얼굴 검출·임베딩/실사성 판단. 실패 검출·신뢰도 부족은 억지 점수 0이 아니라 **N/A + 사유 코드**로 기록하고 가림/작은 얼굴은 수동 검토.

**단계 5 · 블라인드 사람 평가:** 파이프라인 이름을 숨긴 4-way 비교. 컷별 세 시드 중 사전에 고정된 룰(예 종합점수 최대, 동점일 경우 seed가 작은 것)로 한 장만 택해 사람 평가. 2명 독립 평가로 1~5점 매긴다.

### 7.1 권장 QA 지표 (원점수·분모·적용 대상 분리)

| 항목 | 계산/평가 | 추천 점검 기준 | 주의 |
|---|---|---|---|
| **Pose PCK@0.1** | 사람이 검수한 GT 관절과 생성 관절 간 2D 거리, bbox 대각선 10% 이내 비율 | ≥0.75 **임시 시작점** | 어려운 구도/가림 분리. GT 웹툰 추정치는 수동 교정 후 사용 |
| **BBox composition** | 주요 인물 bbox 중심 거리, 너비·높이 비율, 영역 IoU | IoU ≥0.60 **임시** | 얼굴 컷·전신 컷 별로 다른 허용오차 적용 |
| **Identity cosine** | 정규화된 ArcFace/AdaFace 유사 계열 임베딩 유사도 | 절대 통과값 사전 설정 금지; 같은 인물 양성/타인 음성 세트로 보정 | 작고 가려진 얼굴, 측면에 취약; 임베딩 모델/가중치 라이선스 별도 검토 |
| **Identity across cuts** | 같은 `person_id`의 생성 이미지 쌍별 코사인 평균/분산 | 동일인 intra 평균↑ 분산↓ | 표정/각도 다양성 유지하는지 함께 검사 |
| **Photorealism 1~5** | 평가자 2명 블라인드: 피부·광원·사진 질감·그림체 잔존 | 평균 ≥4/5 **임시** | CLIP 점수만으로 사진성 단정 금지 |
| **Anatomy 1~5** | 손가락 수, 팔·다리 연결, 비정상 변형 | 평균 ≥4/5 **임시** | 자동 VLM은 보조, 최종 사람 검토 |
| **Style leakage 0~3** | 0=없음, 1=부분적, 2=뚜렷, 3=대부분 그림 | ≤1 임시 | 셀 셰이딩·윤곽선·카툰 눈 모양 별도 기록 |
| **Operational** | stage별 wall time, cold/warm, peak allocated VRAM, OOM count, retries | 성공률↑ 시간↓ | 모델 reload 포함/제외 시간을 분리 |

**결정 원칙:** 단일 종합 점수만으로 순위를 정하지 말고 `identity`, `pose`, `photorealism`, `failure_rate`, `runtime` 각각 Pareto 비교를 제공한다. 필요하다면 내부 참고용 가중합 `35% identity + 25% pose + 25% realism + 15% composition`을 사용하되, **점수를 0~1로 정규화하고 N/A 처리를 명시**한다. 이 비율은 연구자가 정한 초기 가정이며 공개 벤치마크의 표준이 아니다.

### 7.2 실패 코드 및 재시도 정책

- `POSE_FAIL`: 팔/다리 자세 이탈 → pose 추출/수동 교정 단계 재확인.
- `IDENTITY_FAIL`: 다른 얼굴/머리 → 참조 사진 선택·BFS 입력 순서·마스크 재확인.
- `STYLE_LEAK`: 만화 선/셀 셰이딩 → B의 realism 단계 우선 분석.
- `ANATOMY_FAIL`: 손·얼굴·비정상 연결 → 해당 부위 재생성/인페인팅(본 비교에서는 별도 rescue 실험으로 격리).
- `BG_DRIFT`: 배경/구도 변화 → 원본 bbox/Depth 조건 확인.
- `FACE_UNSCORABLE`: 얼굴이 작거나 가림 → 자동 identity N/A + 사람 평가로 넘김.
- `OOM`, `MISSING_NODE`, `MISSING_WEIGHT`, `LICENSE_BLOCK`: 실행 불가 요인. 품질 점수에 섞지 말고 운영 지표로 표기.

**재시도:** 본실험 1차 결과에는 자동 재시도를 포함하지 않는다. 실패 원인별 재실행은 `rescue/`로 분리해 **재생성 횟수와 비용**을 추가 측정한다. 그렇지 않으면 특정 모델만 무제한 보정해 공정성을 잃는다.

## 8. 실험 로그/QA CSV 입력 규칙

ZIP에 아래 파일 템플릿 포함:

- `data/cuts_manifest.csv`: 20컷 기본 분할과 주석 컬럼; 실제 `comic_path`, `person_id`, `reference_dir` 기입.
- `data/runs_template.csv`: arm × cut × seed의 **240개 행**; stage/모델/시간/VRAM/status을 추적.
- `data/qa_scores_template.csv`: 240개 행에 자동 QA·평가자 점수 빈칸 포함.
- `data/config_sweep.json`: arm별 공식 권장값과 실험 초기 가정(추후 확정)을 분리.
- `data/README_DATA.md`: CSV 필드 및 실행 순서.

추천 산출표: `arm별 성공률`, `PCK`, `정체성 유사도(평가 가능 컷 수와 함께)`, `실사성 평균`, `인체 이상 비율`, `컷당 총 생성 시간`, `재시도 후 승인 비율`.

## 9. 프롬프트/연결 체크리스트

**공통 장면 설명(예시):**

```text
Live-action editorial photograph. Maintain the pose, camera angle, subject placement,
hand gesture, gaze and clothing category shown in the structural guide.
Natural skin texture and pores, physically plausible shadows, believable lighting,
realistic hair strands, coherent facial anatomy. No cartoon line art, cel shading,
illustration outlines or plastic CGI skin.
```

**B V3:**

```text
transform the image to realistic photograph. Keep the scene composition,
pose, emotion and wardrobe silhouette. Real photographic skin and lighting.
[Experimental dual-image] with the character looking exactly like the one in Figure 2.
```

**C Pose:** pose 지도 `image1`, 인물/의상/조명은 텍스트로 묘사. 다음 BFS Krea head 단계에서 `image1=목표 장면`, `image2=참조 얼굴`, 예시 `head_swap: replace the head with the reference head.`

**D Fisher:** Fisher 최신 통합 인코더의 이미지 배열 순서는 **현재 공식 JSON 설정**에서 확인; VNCCS 단독 약식 예시를 복사할 때 이미지 순서 혼동 금지. 다음 BFS Qwen head 단계에서는 `image1=생성된 장면`, `image2=인물 참조`, `head_swap` 설명 사용.

## 10. 1차 합격 기준 및 후속 실험

**실무 PoC 합격(임시):** 20컷 중 최소 16컷에서 큰 포즈/인체 문제 없이 후보 1개 이상 확보, 얼굴 평가 가능한 컷의 identity 성능이 A 대비 개선, 실사성 평가 평균 4/5 이상, GPU OOM 최소화. 정확한 숫자는 스모크 결과와 사업 목표에 맞춰 수정한다.

**후속 소거 실험:**
1. C: Pose only vs Pose+BFS Krea → BFS의 효과 분리.
2. D: Fisher only vs Fisher+BFS Qwen → identity 개선 대비 구도 손상 측정.
3. B: V3 단독 vs V3 + Dual-Image identity → 참조 이미지 효과 측정.
4. 모든 arm: 얼굴 클로즈업 vs 전신의 유사도 하락 분리.

**권장 우선순위:** C가 가장 직접적으로 현재 포즈·얼굴 제어 문제를 개선할 수 있는 구성; B는 실사 스타일과 원본 스타일 잔존의 기준선; D는 복잡한 포즈 검증; A는 현재 시스템 대비 개선율 산출. 결과는 반드시 실측 후 판단한다.

## 11. 사용권·개인정보·재현성 주의

- **Qwen:** [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE)는 비상업적 연구 사용만 허용하고 상업적 사용은 별도 계약 요구.
- **Krea:** [Krea 2 Turbo 게이트 페이지](https://huggingface.co/krea/Krea-2-Turbo)에 표기된 Krea 2 Community License와 Acceptable Use Policy 수락 후 이용. 회사 규모·상업 사용·콘텐츠 제한을 최신 원문으로 검토.
- **LoRA:** LoRA 저장소의 MIT/Apache는 기반 가중치의 제한을 없애지 않는다. 학습/출력물의 권리와 레퍼런스 인물의 동의를 별도 관리.
- **성능 주장:** 이 계획은 재현 가능한 실험 설계이지, 로컬 GPU에서 네 조합의 성공 구동을 확인한 결과 보고서가 아니다.

## 12. 공식 링크 / 확인 일자

- [Comfy-Org Krea2 패키징 및 파일명](https://huggingface.co/Comfy-Org/Krea-2) — 확인 2026-10-15
- [Krea2 Anything2RealCharacters V3 권장 Steps 8 / CFG 1](https://huggingface.co/WarmBloodAban/Krea2_Anything2RealCharacters) — 확인 2026-10-15
- [Krea Pose LoRA / Ostris 인코딩 / Steps ~10](https://huggingface.co/thedeoxen/Krea-2-pose-controlnet) — 확인 2026-10-15
- [Ostris Krea Edit ComfyUI 노드](https://github.com/ostris/ComfyUI-Krea2-Ostris-Edit) — 확인 2026-10-15
- [BFS Krea head/body 안내](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/blob/main/docs/krea-2.md) — 확인 2026-10-15
- [BFS Qwen 2.1 head 안내](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/blob/main/docs/qwen-image-2.1.md) — 확인 2026-10-15
- [Comfy-Org Qwen-Image-2.1 모델](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) — 확인 2026-10-15
- [Fisher Pose 설치 및 최신 워크플로우](https://github.com/Work-Fisher/ComfyUI-Fisher-Pose) — 확인 2026-10-15
- [VNCCS PoseStudio 모델](https://huggingface.co/MIUProject/VNCCS_PoseStudio_QI2.1) — 확인 2026-10-15
- [comfyui_controlnet_aux DWPose](https://github.com/Fannovel16/comfyui_controlnet_aux) — 확인 2026-10-15
