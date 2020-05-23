# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

We need to query the network using query_network() method which returns the names of all supported layers that the IR loaded network have. Then, we match these supported layers with all of net's layers(using net.layers.keys()) and all the unmatching layers are termed as unsupported layers. And any unsupported layers are automatically classified as the custom layers by the Model Optimizer. The process behind converting custom layers model varies depending on the model framework one is using. The first step is to register the custom layers as extensions to the Model Optimizer, so we generate the extension template files for custom layers using model extension generator. Then, the second step for tensor flow would be to actually replace the unsupported subgraph with a different subgraph. The final TensorFlow option is to actually offload the computation of the subgraph back to TensorFlow during inference, whereas for caffe model the second step is to register the layers as Custom, then use Caffe to calculate the output shape of the layer. You’ll need Caffe on your system to do this option. Hence, we convert the custom layers into supported layers. Apart from this we can also use software specific extensions for adding extra support to the model's layers, like if the device is CPU then, we could add CPU specific extension while loading the network.

We need to handle the custom layers as they are not included into a list of known layers, and when model optimizer marked them as custom ones, then Inference Engine would not know what to do which these custom layers. So, we need to avoid those errors by adding extra support.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations were as follows:
1.For pre-coversion model - I used the statistics mentioned on TF model zoo @ https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md . I tried to write a script to test it in the workspace environment as that comparison would have been more accurate as it would have been utilizing exactly the hardware, but I could not install/satisfy all the requirements to run tf there. I have just started learning deep learning recently so I am using online resources to get an estimate of SSD Mobilenet V2 coco tensorflow model's accuracy and inference time.
2.For post-conversion model - I implemented the code to calculate the inference time per frame using time module and found that IR model was taking about 70ms per frame on avaerage. And for calculating the accuracy on the given video, I used an counter and increased its value for every frame where the model was inferencing that there is a person in the frame and divided it the with the total number of frames in which there was actually a person in the frame (I have counted seconds manually when a person was actually visible on the frame, and have multiplied them with the fps value extracted from VideoCapture). I found that post-conversion model accurately found the person in frame with 70.444% times while using 0.6 as the conf threshold.

The difference between model accuracy pre- and post-conversion was lesser than 2 percent.

The size of the model pre- and post-conversion was 66.4 MB and 64.1 MB respectively(I am reporting the file size of .pb (Tensorflow) and .bin (IR).).

The inference time of the model pre- and post-conversion was 31ms and 70.444ms respectively.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are at the shopping mall's buying counters, flight boarding queues, voting campaigns, public CCTVs where social distancing needs to be maintained, ATM counters, surveillance without privacy loss, etc.

Each of these use cases would be useful because we can count the number of people in each and every frame, so we could raise an alarm if more than certain people are on the frame where distance or queue needs to be maintained. As the app does not check the person features so it could adhere to the privacy rules of people while counting the total number of people crossed per day(and, if object tracker is added then we can also tell that how many people entered and exited from which side). Also, as it counts the duration of a person on a screen it can automatically tell people if they are breaking any time limit alloted to them.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows:
1.Image size would not effect much as we would resize and only then we will feed it to the model as per model's requirement, though if the input image's original size is way out of order than the model's requirements, then the resize function could distort it very much as it would have to interpolate it quite much and the resulting input image if very distorted can result in low accuracy.
2.Lighting can also effect the model's predictions as interpretation and confidence  intervals of the bounding boxes would fall if very low or very high lighting. 
3.Model's accuracy has also significant effect on the edge model application's accuracy, as in Edge the model optimizer optimizes the model by sacrificing some of the its accuracy, so if a model already has low accuracy than we should expect that the edge application would have even lower accuracy.

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
