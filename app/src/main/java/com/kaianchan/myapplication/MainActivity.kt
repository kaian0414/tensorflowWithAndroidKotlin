package com.kaianchan.myapplication

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import kotlinx.android.synthetic.main.activity_main.*
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import org.tensorflow.lite.Interpreter

class MainActivity : AppCompatActivity() {

    // Use the function and show the result
    private lateinit var interpreter: Interpreter


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)


        interpreter = Interpreter(loadModelFile(),null)


        btn_predict.setOnClickListener {
            val str = et_input.text.toString().trim()
            if (str.isNotEmpty()){
                tv_result.text = "result : ${doInference(str)}"
            }
        }
    }

    // load the model
    private fun loadModelFile():MappedByteBuffer {
        val assetFileDescriptor = this.assets.openFd("linear.tflite")
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffSet = assetFileDescriptor.startOffset
        val length = assetFileDescriptor.length

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffSet,length)
    }

    // Predict
    private fun doInference(str:String):Float{
        val input = FloatArray(1)
        input[0] = str.toFloat()


        val output = Array(1) {FloatArray(1)}
        interpreter.run(input,output)
        return output[0][0]
    }



}
