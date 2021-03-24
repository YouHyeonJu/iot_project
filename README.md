# 라즈베리파이4 를 이용한 코로나 열감지,마스크 감지 프로그램
led,buzzer,Thermal Imaging Camera(TIC),pi_Camera



## 열화상 카메라 설정 방법
- 예제와 설정(아래 링크)  
[![열 화상 카메라 ](https://img.youtube.com/vi/rqdTx0AKroE/0.jpg)](https://youtu.be/rqdTx0AKroE?t=172)

### 핀 번호
- buzzer=18
- red=14
- green=15

## 마스크 착용여부 감지 모델
- [achonyws님의 깃허브](https://github.com/gachonyws/face-mask-detector)에 있는 face_mask_detector 모델을 가져와 목적에 맞춰 수정
