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
    TfLiteModel * model;
    TfLiteInterpreterOptions* options;
    TfLiteInterpreter * interpreter;
    TfLiteTensor * input_tensor;
    const TfLiteTensor * output_tensor;
    FullDuplexPass() {
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
        // Copy the input samples to the output with a little arbitrary gain change.

        // This code assumes the data format for both streams is Float.
        const float *inputFloats = static_cast<const float *>(inputData);
        float *outputFloats = static_cast<float *>(outputData);

        // It also assumes the channel count for each stream is the same.
        int32_t samplesPerFrame = outputStream->getChannelCount();
        int32_t numInputSamples = numInputFrames * samplesPerFrame;
        int32_t numOutputSamples = numOutputFrames * samplesPerFrame;
       // float l;
      //  float N;
       // int32_t samplesToProcess = std::min(numInputSamples, numOutputSamples);
        // It is possible that there may be fewer input than output samples.
//        for (int32_t i = 0; i < samplesToProcess; i++) {
//            l = static_cast<float>(i);
//            N = static_cast<float>(samplesToProcess);
//            ALOG("float: %f", *inputFloats);
//            *outputFloats++ = *inputFloats++ * sin(2.0 * l * M_PI / N); // do some arbitrary processing
//        }
            ALOG("float: %f", *inputFloats);
        TfLiteTensorCopyFromBuffer(
                input_tensor,
                inputFloats,
                TfLiteTensorByteSize(input_tensor));
        TfLiteInterpreterInvoke(interpreter);
        TfLiteTensorCopyToBuffer(
                output_tensor,
                outputFloats,
                TfLiteTensorByteSize(output_tensor));


        // If there are fewer input samples then clear the rest of the buffer.
        int32_t samplesLeft = numOutputSamples - numInputSamples;
        for (int32_t i = 0; i < samplesLeft; i++) {
            *outputFloats++ = 0.0; // silence
        }

        return oboe::DataCallbackResult::Continue;
    }
};
#endif //SAMPLES_FULLDUPLEXPASS_H
