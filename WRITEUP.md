# WRITEUP

## Explaining Custom Layers

The process behind converting custom layers involves multiple methods. The most common way to do so is to register the custom layer as an extension and add it when the Model Optimizer converts the model to an IR. Depending on the original framework of the model, it is also possible to offload the custom layer's computation to the original framework. However, doing so might increase the inference time.

The main reason for handling custom layers is that the OpenVINO Model Optimizer doesn't support all layers from every framework. Moreover, being able to run individual layers outside of the Inference Engine can be used to debug models layer by layer.

## Comparing Model Performance

My methods to compare models before and after conversion to Intermediate Representations was first to look at the detection results on the test video provided. It would give me a first appreciation of the inference time and the accuracy of the model. Then I used the tools provided in [rafaelpadilla/Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics). As the model was trained on the Pascal VOC dataset, I wanted to measure its performances on the same dataset. I used for this the test set from VOC 2007:

| Model    | Size (MB) | Inference Time (s) | Accuracy (mAP) | Person Accuracy (AP) |
|----------|-----------|--------------------|----------------|----------------------|
| Pytorch  | 105.15    | 0.463              | 64.67%         | 61.87%               |
| OpenVINO | 102.9     | 0.289              | 66.51%         | 63.68%               |  

The detailed results of those test can be found here: [link](https://github.com/LucasVandroux/ssd.pytorch/releases/tag/v0.9.1)

It is interesting to note that on this specific dataset, the OpenVINO model scores a little bit higher than the original PyTorch model. It might be due to the testing method.

## Differences between Cloud and Edge deployments

There are many differences in network needs and costs using cloud services instead of deploying at the edge. First, if using the cloud, all the images need to be sent to the cloud for inference. It can have a considerable impact on the network. Moreover, using the cloud adds some latency caused by the data transfer. On the other hand, when deployed at the edge, everything is done on the device. Finally, depending on the need of the model, the one time cost for the hardware to make the inference at the edge can be significantly lower than the cloud cost in the long term. The security of the data is another criterion to consider, indeed transferring data to and from the cloud can be a significant security risk.

## Assess Model Use Cases

Using the people counter app, one can understand the flows of people entering and exiting a place or monitor how many people are at the same time in a location. 

For example, in a shop, it might be interesting to know when people are entering and exiting and compare it to the sales to get market insights. Moreover, with the current pandemic, we must make sure that we are keeping the distance between each other. Therefore, it is necessary to control in real-time the number of people in a place to avoid this place to be over-crowed.

## Assess Effects on End-User Needs

Lighting, camera focal length, and image size, if too different from the data the model was trained on, can damage the accuracy of the model. In the end, a reduction of the accuracy of the model makes the statistics extracted with it less precise and ultimately less useful. 

## Model Research

As I am mainly working with PyTorch, I wanted to find a model made with PyTorch and convert it to ONNX before using the Model Optimizer. After searching on GitHub, I decided to use the SSD model from https://github.com/amdegroot/ssd.pytorch. I used the following command to convert the ONNX representation to an IR:

```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo_onnx.py --input_model ssd.onnx --mean_values [123,117,104] --reverse_input_channels --input_shape [1,3,300,300]
```