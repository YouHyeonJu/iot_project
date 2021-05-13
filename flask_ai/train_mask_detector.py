# import packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to input dataset")
ap.add_argument("-p","--plot",type=str,default="plot.png",help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# learning rate, epochs, batch size 설정
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# 이미지가 있는 폴더 가져오기
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# 이미지 경로 반복하기
for imagePath in imagePaths:
    # 파일 이름으로 클래스 라벨 추출
    label = imagePath.split(os.path.sep)[-2]

    # 인풋 이미지를 (224x224) 사이즈로 불러오고 전처리
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    # 데이터, 라벨 리스트 업데이트
    data.append(image)
    labels.append(label)

# 데이터, 라벨 리스트를 넘파이 배열로 변환
data = np.array(data, dtype="float32")
labels = np.array(labels)

# 라벨에 one-hot 인코딩 실행
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# 데이터를 75%를 학습용, 25%를 테스팅용으로 분리
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.20,stratify=labels,random_state=42)

# 데이터 확대를 위한 학습용 이미지 생성기를 구성
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest')

# MobileNetV2 net 불러오기
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))

# 기본 모델위에 올릴 해드 모델 구조 구성하기
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# 기본 모델 위에 헤드 모델을 배치(훈련할 실제 모델)
model = Model(inputs=baseModel.input, outputs=headModel)

# 뭔지 모르겠음
for layer in baseModel.layers:
    layer.trainable = False

# 모델 컴파일
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(loss='binary_crossentropy',optimaizer=opt,metrics=['accuracy'])

# net의 해드부분 학습
print("[INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX,testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# 테스팅 데이터 셋으로 모델 평가
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# 각 테스팅 테이터의 예측 확률이 가장 큰 레이블 인덱스 찾기
predIdxs = np.argmax(predIdxs, axis=1)

# 분류 보고서 출력
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# 모델 저장
print('[INFO] saving mask detector model...')
tf.saved_model.save(model,'mask_detector')

# 학습 손실과 정확도 plot
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])