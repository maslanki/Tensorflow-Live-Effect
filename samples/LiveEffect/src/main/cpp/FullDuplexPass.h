/*
 * Copyright 2018 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SAMPLES_FULLDUPLEXPASS_H
#define SAMPLES_FULLDUPLEXPASS_H
#define  LOG_TAG    "testjni"
#define  ALOG(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)

#include "../jni/c_api.h"
#include "../jni/common.h"
#include "FullDuplexStream.h"

class FullDuplexPass : public FullDuplexStream {
public:

    float* tempSamplesArray1;
    float* tempSamplesArray2;
    float* tempSamplesArray3;

    int iterator;

    TfLiteModel * model;
    TfLiteInterpreterOptions* options;
    TfLiteInterpreter * interpreter;
    TfLiteTensor * input_tensor;
    const TfLiteTensor * output_tensor;

    FullDuplexPass() {
        tempSamplesArray1 = (float *) malloc(64 * sizeof(float));
        tempSamplesArray2 = (float *) malloc(64 * sizeof(float));
        tempSamplesArray3 = (float *) malloc(64 * sizeof(float));
        iterator = 0;

        model = TfLiteModelCreateFromFile("/data/data/com.google.oboe.samples.liveeffect/model.tflite");
        options = TfLiteInterpreterOptionsCreate();
        interpreter = TfLiteInterpreterCreate(model, options);
        TfLiteInterpreterAllocateTensors(interpreter);
        input_tensor =
                TfLiteInterpreterGetInputTensor(interpreter, 0);
        output_tensor =
                TfLiteInterpreterGetOutputTensor(interpreter, 0);

    }
    virtual oboe::DataCallbackResult
    onBothStreamsReady(
            std::shared_ptr<oboe::AudioStream> inputStream,
            const void *inputData,
            int   numInputFrames,
            std::shared_ptr<oboe::AudioStream> outputStream,
            void *outputData,
            int   numOutputFrames) {

        const float *inputFloats = static_cast<const float *>(inputData);
        float *outputFloats = static_cast<float *>(outputData);

        //4 x 64 samples to process
        float *total = (float *) malloc(256 * sizeof(float));
        memcpy(total, tempSamplesArray1, 64 * sizeof(float));
        memcpy(total + 64, tempSamplesArray2, 64 * sizeof(float));
        memcpy(total + 128, tempSamplesArray3, 64 * sizeof(float));
        memcpy(total + 192, inputFloats, 64 * sizeof(float));

//        ALOG("ITERATION: %d", ++iterator);
//        for (int32_t i = 0; i < 256; i++) {
//            ALOG("%d. LAST total %f",i, total[i] );
//        }

        int32_t samplesPerFrame = outputStream->getChannelCount();
        int32_t numInputSamples = numInputFrames * samplesPerFrame;
        int32_t numOutputSamples = numOutputFrames * samplesPerFrame;

        TfLiteTensorCopyFromBuffer(
                input_tensor,
                total,
                TfLiteTensorByteSize(input_tensor));
        TfLiteInterpreterInvoke(interpreter);
        TfLiteTensorCopyToBuffer(
                output_tensor,
                outputFloats,
                TfLiteTensorByteSize(output_tensor));


        int32_t samplesToProcess = std::min(numInputSamples, numOutputSamples);
        ALOG("num of samples: %d", samplesToProcess);

        memcpy(tempSamplesArray1, tempSamplesArray2, 64 * sizeof(float));
        memcpy(tempSamplesArray2, tempSamplesArray3, 64 * sizeof(float));
        memcpy(tempSamplesArray3, inputFloats, 64 * sizeof(float));

//        for (int32_t i = 0; i < 64; i++) {
//
//          //  ALOG("%d: inputs: %f", i, inputFloats[i]);
//            *outputFloats++ *= 10;
//        }

        // If there are fewer input samples then clear the rest of the buffer.
        int32_t samplesLeft = numOutputSamples - numInputSamples;
        for (int32_t i = 0; i < samplesLeft; i++) {
            *outputFloats++ = 0.0; // silence
        }
        return oboe::DataCallbackResult::Continue;
    }
};
#endif //SAMPLES_FULLDUPLEXPASS_H
