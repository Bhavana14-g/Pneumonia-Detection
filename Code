from google.colab import drive
drive.mount('/content/drive/')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import tensorflow as tf

from cv2 import imshow
from google.colab.patches import cv2_imshow
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.callbacks import Callback,EarlyStopping
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import classification_report
from imutils import paths
b='/content/drive/MyDrive/dataset/NORMAL/IM-0001-0001.jpeg'
a=cv2.imread(b)
cv2_imshow(a)
def get_label(path:str) -> str:
  return path.split("/")[-2]

labels = list(map(lambda x : get_label(x) , file_paths))
print(labels)
def get_label(path:str) -> str:
  return path.split("/")[-2]

labels = list(map(lambda x : get_label(x) , file_paths))
print(labels)
counts=data.Label.value_counts()
sns.barplot(x=counts.index,y=counts)
plt.xlabel('Type')
plt.xticks(rotation=90)
train,test=train_test_split(data,test_size=0.25,random_state=42)
fig,axes=plt.subplots(nrows=5,ncols=3,figsize=(10,8),subplot_kw={'xticks':[],'yticks':[]})
for i,ax in enumerate(axes.flat):
  ax.imshow(plt.imread(data.Filepath[i]))
  ax.set_title(data.Label[i])
plt.tight_layout()
plt.show()
datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
train_gen=datagen.flow_from_dataframe(
    dataframe=train,
    x_col='Filepath',
    y_col='Label',
    target_size=(224,224),
    class_mode ='categorical',
    batch_size=32,
    shuffle=True,
    seed=42
)
val_gen=datagen.flow_from_dataframe(
    dataframe=test,
    x_col='Filepath',
    y_col='Label',
    target_size=(224,224),
    class_mode='categorical',
    batch_size=32,
    shuffle=False,
    seed=42
)
test_gen=datagen.flow_from_dataframe(
    dataframe=test,
    x_col='Filepath',
    y_col='Label',
    target_size=(224,224),
    class_mode='categorical',
    batch_size=32,
    shuffle=False,
)
from tensorflow.keras.applications import ResNet50
pretrained_model=ResNet50(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
pretrained_model.trainable=False
inputs=pretrained_model.input
x=Dense(128,activation='relu')(pretrained_model.output)
x=Dense(128,activation='relu')(x)
outputs=Dense(2,activation='softmax')(x)
model=Model(inputs=inputs,outputs=outputs)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
my_callbacks=[EarlyStopping(monitor='val_accuracy',
                            min_delta=0,
                            patience=2,
                            mode='auto')]
                            history=model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=my_callbacks
)
plt.figure(figsize=(5,3))
pd.DataFrame(history.history)[['accuracy','val_accuracy']].plot()
plt.title('Accuracy')
plt.show()
print()
plt.figure(figsize=(7,3))
pd.DataFrame(history.history)[['loss','val_loss']].plot()
plt.title('Loss')
plt.show()
results=model.evaluate(test_gen,verbose=0)

print('Test Loss: {:.5f}'.format(results[0]))
print('Test Accuracy: {:.2f}'.format(results[1]*100))
pred=model.predict(test_gen)
pred=np.argmax(pred,axis=1)

labels=(train_gen.class_indices)
labels=dict((v,k) for k,v in labels.items())
pred=[labels[k] for k in pred]
y_test=list(test.Label)
print(classification_report(y_test,pred))
fig,axes=plt.subplots(nrows=5,ncols=2,figsize=(12,8),
                      subplot_kw={'xticks':[],'yticks':[]})
for i ,ax in enumerate(axes.flat):
  ax.imshow(plt.imread(test.Filepath.iloc[i]))
  ax.set_title(f"True:{test.Label.iloc[i]}\nPredicted:{pred[i]}")
plt.tight_layout()
plt.show()
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input

img_path='/content/drive/MyDrive/dataset/NORMAL/IM-0085-0001.jpeg'
img=cv2.imread(img_path)
img=cv2.resize(img,(224,224))
#cv2.imshow(img)

x=np.expand_dims(img,axis=0)
x=preprocess_input(x)
result=model.predict(x)
print((result*100).astype('int'))
if(result[0][0]>result[0][1]):
  print("NORMAL")
else:
  print("PNEUMONIA")
