# tensorflowWithAndroidKotlin
Ref: 使用TensorFlow在Android App中進行機器學習入門 https://blog.csdn.net/weixin_26739079/article/details/108515540

# --以下是在Google Colab的代碼--
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow import lite

# for numpy training
#y = 2x - 1
x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0])

#Create model with Keras Sequential
model = tf.keras.models.Sequential([
     keras.layers.Dense(units=1, input_shape=[1]),
     keras.layers.Dense(units=1, input_shape=[1])
])

# Compile model
model.compile(
    optimizer = "sgd", # 用於最小化損失函數
    loss = "mean_squared_error" # 預測與實際差多少
)

# Perform training
model.fit(x, y, epochs=15000)
print(model.predict([10])) # y=2x-1=2*10-1=19

#為了進行訓練，我們將使用一個函數，函數將包含特徵(x，(y)和紀元= 50)0次(循環次數)。對於50次進料，將執行前進和後退，獎勵給我們訓練後，我們將測試模型以預測輸出。為10，它給我輸出18.999992，根據我們的模型，實際值應為19(2 * 10–1)。

# Save as Keras format
keras_file = "linear.h5"
tf.keras.models.save_model(
    model,
    keras_file
)

# Change model to tflite
converter = lite.TFLiteConverter.from_keras_model(model)
tfmodel = converter.convert()
open("linear.tflite", "wb").write(tfmodel)

#tflite就是android中需要用的

# --計算y=2x-1, Android App截圖--
![image](https://github.com/kaian0414/tensorflowWithAndroidKotlin/blob/master/tensorflowWithAndroidKotlin.png)
