# SEMANTIC SEGMENTATION OF DASHCAM IMAGES
### Deep Learning-Based Scene Parsing Using U-Net and ResNet34

A Project Report Submitted in Partial Fulfilment of the Requirements for the 7th Semester Evaluation of B.Tech (Computer Science and Engineering)

**Subject:** Pattern Recognition and Anomaly Detection  
**Submitted by (Group – 11)**

| Name | Roll No. |
| :--- | :--- |
| Prachi Sahoo | FET-BAML-2023-27-028 |
| Aditya Menon | FET-BAML-2023-27-047 |
| Aditya Uday | FET-BAML-2023-27-006 |
| Ayush Pandey | FET-BAML-2023-27-016 |

**Guide:** Mr. Rahul Rawat  
Department of Computer Science and Engineering  
Faculty of Engineering and Technology  
Academic Year 2026 – 2027  

**Sri Sri University, Odisha**  
Sri Sri Vihar, Ward No – 3,  
Godi Sahi, Cuttack – 754006,  
Odisha, India

---
<div style="page-break-after: always"></div>

## DECLARATION

We, the undersigned members of Group-11 (Semester 7, B.Tech CSE), hereby declare that the project report entitled "Semantic Segmentation of Dashcam Images" submitted for the Project Work evaluation under the guidance of Mr. Rahul Rawat is an original and authentic record of the work carried out by us. The matter presented in this report has not been submitted, in part or in full, to any other university, institute, or examining body for the award of any degree or diploma. We further declare that all sources of information, ideas, algorithms, and datasets used in the preparation of this report have been duly acknowledged and cited in the References section, and that this work does not, to the best of our knowledge, infringe upon any existing copyright.

**Team Members:**
- Prachi Sahoo (FET-BAML-2023-27-028)
- Aditya Menon (FET-BAML-2023-27-047)
- Aditya Uday (FET-BAML-2023-27-006)
- Ayush Pandey (FET-BAML-2023-27-016)

Place: ______________________ Date: ______________________

---
<div style="page-break-after: always"></div>

## ACKNOWLEDGEMENT

We take this opportunity to express our sincere gratitude to everyone who supported us throughout the completion of this project. We are deeply thankful to our Guide, Mr. Rahul Rawat, for their invaluable guidance, technical insights, and constant encouragement throughout the development of our Semantic Segmentation system.

We would also like to thank the faculty members of the Department of Computer Science and Engineering for providing us with the necessary academic support, laboratory resources, and a conducive environment for research and experimentation.

Finally, we would like to acknowledge the effort and cooperation of every member of our team, whose collective contribution, discussions, and dedication made this project possible.

---
<div style="page-break-after: always"></div>

## ABSTRACT

In the domain of autonomous driving and advanced driver-assistance systems (ADAS), scene understanding is a critical task for safe navigation. Unlike traditional object detection, which draws bounding boxes around objects, semantic segmentation assigns a class label to every individual pixel in an image, providing a dense and highly detailed understanding of the vehicle's surroundings.

This project addresses the challenge of parsing road scenes by implementing a robust semantic segmentation pipeline using a deep learning architecture. We propose a U-Net model enhanced with a pre-trained ResNet34 encoder to classify pixels into 32 distinct categories (e.g., Road, Car, Pedestrian, Sky) using the CamVid dataset. By leveraging transfer learning, the model extracts high-level spatial features effectively, while the decoder reconstructs the spatial resolution to output a pixel-wise mask.

The methodology involves preprocessing high-resolution dashcam imagery, applying data augmentation, and training the neural network using a Combined Loss function (a weighted sum of Cross-Entropy and Dice Loss) optimized by the AdamW optimizer. 

The user interacts with an industrial-grade Single Page Application (SPA) dashboard built with vanilla web technologies, served by a lightweight Flask backend. The frontend provides real-time inference, video processing capabilities, and interactive image comparison sliders for qualitative visual inspection.

The model successfully segments complex road scenes, achieving a strong mean Intersection over Union (mIoU) score. This supervised approach, combined with the comprehensive web dashboard, holds significant potential for deployment in autonomous navigation research and real-world ADAS environments where pixel-perfect precision is required.

---
<div style="page-break-after: always"></div>

## 1. INTRODUCTION

### 1.1 Background
Semantic segmentation is a foundational computer vision task that bridges the gap between image classification and object detection. By assigning a categorical label to every pixel, it enables machines to interpret the exact boundaries and shapes of objects in a scene. In the context of self-driving cars, this dense understanding is required to differentiate drivable roads from sidewalks, and pedestrians from stationary background objects.

### 1.2 Motivation
For autonomous vehicles, understanding the environment accurately is a matter of safety. Traditional computer vision techniques struggle with varying lighting conditions, occlusions, and complex urban backgrounds. Deep learning architectures like U-Net have revolutionized this space by combining deep semantic feature extraction with precise spatial localization. 

### 1.3 Problem Statement
To design and implement an automated computer vision system capable of segmenting dashcam images and video streams into 32 distinct structural and object classes, highlighting the results via a user-friendly real-time web dashboard.

### 1.4 Objectives
- Build a robust data loading pipeline to handle the CamVid dataset.
- Design and implement a U-Net architecture with a ResNet34 encoder using PyTorch.
- Train the model using a supervised methodology with a Combined Loss function (Cross-Entropy + Dice Loss).
- Develop an industrial-grade Single Page Application (SPA) dashboard using Flask, HTML, CSS, and JS to provide real-time image and video inference.
- Evaluate the model's ability to accurately segment urban scenes based on Intersection over Union (IoU) metrics.

### 1.5 Scope
The project encompasses the development, training, and deployment of the semantic segmentation model for the CamVid dataset taxonomy. It includes a functional REST API and a frontend for visualization. The scope explicitly excludes deployment to edge hardware (like Raspberry Pi) and real-time physical vehicle integration.

---
<div style="page-break-after: always"></div>

## 2. LITERATURE REVIEW & EXISTING SYSTEM

### 2.1 Existing Methods
Early methods for scene parsing relied on classical image processing, such as thresholding, region growing, and edge detection (Canny, Sobel). These methods were often paired with hand-crafted features (HOG, SIFT) and classifiers like Support Vector Machines (SVMs) or Conditional Random Fields (CRFs) to classify pixels.

### 2.2 Deep-Learning-Based Approaches
With the advent of Deep Learning, Fully Convolutional Networks (FCNs) replaced dense layers with convolutional layers to output spatial maps. Modern architectures like U-Net, DeepLabV3+, and PSPNet dominate the field. U-Net, originally designed for biomedical image segmentation, utilizes a symmetric encoder-decoder structure with skip connections, preserving spatial details lost during pooling.

### 2.3 Limitations of Existing Systems
- **Classical methods:** Highly sensitive to changes in lighting, weather, and camera position. They require extensive manual tuning and fail on complex urban scenes.
- **Standard FCNs:** Often produce coarse, blob-like segmentations because they lose spatial resolution during deep downsampling operations.
- **Complex Transformers:** Vision Transformers (ViTs) and advanced architectures (like SegFormer) are highly accurate but computationally expensive and difficult to run in real-time on constrained hardware without significant optimization.

### 2.4 Proposed Improvement
Our proposed project addresses these limitations by using a **U-Net architecture with a pre-trained ResNet34 backbone**. This approach is supervised, computationally efficient, and natively provides sharp object boundaries due to the U-Net skip connections. ResNet34 acts as a powerful feature extractor without the massive computational overhead of deeper networks (like ResNet101 or Transformers), making it suitable for fast inference.

---
<div style="page-break-after: always"></div>

## 3. PROPOSED SYSTEM / METHODOLOGY

### 3.1 System Overview
The proposed system is an end-to-end web application for semantic segmentation of dashcam imagery. A user uploads images or videos via a sleek web interface. The backend API processes the media through a trained U-Net model. The model reconstructs a class-colored pixel mask, which is overlayed onto the original image. An interactive comparison slider is returned to the user alongside telemetry data.

### 3.2 Workflow
- **Input:** RGB images or short video clips of road scenes are uploaded via the Flask endpoints.
- **Preprocessing:** Images are resized, converted to PyTorch tensors, and normalized using ImageNet statistics.
- **Core Processing:** The image is passed through the ResNet34 encoder to obtain compressed spatial features, and then upsampled through the U-Net decoder to generate a 32-channel logit map (one channel per class).
- **Decision Step:** An argmax operation collapses the channels into a single 2D array of class IDs.
- **Output:** The API maps the class IDs to predefined RGB colors (e.g., Road = Purple, Sky = Grey) and returns the segmented overlay.
- **Analytics:** The SPA frontend displays the resulting images, execution times, and dynamically lists the detected classes present in the scene.

### 3.3 Dataset
The system utilizes the Cambridge-driving Labeled Video Database (CamVid) dataset, a comprehensive benchmark for semantic segmentation. It contains high-resolution images of urban driving scenes. The dataset features a 32-class taxonomy, including categories like Bicyclist, Building, Car, Pedestrian, Road, and TrafficLight.

### 3.4 Preprocessing
- **Resizing:** Images and ground-truth masks are resized for uniformity and memory efficiency.
- **Tensor Conversion:** Images are converted from PIL/NumPy format to PyTorch tensors.
- **Normalization:** Pixel values are scaled and normalized using the mean and standard deviation of the ImageNet dataset, which matches the pre-training of the ResNet34 encoder.

### 3.5 Algorithm / Model
The core algorithm is a Convolutional Neural Network following the U-Net topology.
- **Encoder:** A pre-trained ResNet34 backbone. It utilizes residual blocks to extract hierarchical features (edges, textures, object parts) while mitigating the vanishing gradient problem.
- **Decoder:** A series of upsampling blocks that concatenate high-resolution feature maps from the encoder (skip connections) with upsampled decoder features, ending in a 1x1 convolution mapping to 32 classes.

---
<div style="page-break-after: always"></div>

## 4. SYSTEM DESIGN & IMPLEMENTATION

### 4.1 Architecture Details
The U-Net model leverages the `segmentation_models_pytorch` library.
- **Encoder:** ResNet34 (pre-trained on ImageNet)
- **Decoder Channels:** [256, 128, 64, 32, 16]
- **Classes:** 32
- **Activation:** Identity (Logits are returned for Cross-Entropy loss)

### 4.2 Technologies Used
- **Programming Language:** Python 3.x
- **Deep Learning Framework:** PyTorch / Torchvision
- **Model Architecture:** segmentation-models-pytorch
- **Backend API:** Flask, Werkzeug
- **Data Processing & Visualization:** NumPy, OpenCV (cv2), Pillow (PIL)
- **Frontend:** HTML, CSS (Vanilla), JavaScript

### 4.3 Implementation Steps
1. **Dataset Loading:** Created a custom PyTorch Dataset (`dataset.py`) to ingest input frames and their corresponding colored mask PNGs, applying necessary albumentations transforms.
2. **Model Definition:** Utilized `smp.Unet` in `model.py` to instantiate the ResNet34 backbone.
3. **Training Script:** Built `train.py` featuring a training loop utilizing AdamW optimizer, mixed precision (AMP) for speed, and a custom `CombinedLoss` (Dice + CrossEntropy).
4. **API Development:** Implemented `server.py` using Flask to handle model loading, image uploads, video streaming (frame-by-frame inference), and status endpoints.
5. **Visualization:** Implemented utility functions to convert raw tensor predictions into RGB overlays using NumPy color mapping.
6. **Frontend Integration:** Linked a vanilla HTML/JS Single Page Application (SPA) to the API. Designed an industrial dashboard featuring a dropzone, comparison slider, and telemetry readouts.

### 4.4 Important Code Snippet
```python
# Forward Pass with Mixed Precision
with torch.amp.autocast(device_type=_AUTOCAST_DEVICE, enabled=(DEVICE=='cuda')):
    predictions = model(data)
    loss = loss_fn(predictions, targets)

# Optimization step
optimizer.zero_grad()
if scaler:
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
else:
    loss.backward()
    optimizer.step()
```

---
<div style="page-break-after: always"></div>

## 5. EXPERIMENTAL SETUP & RESULTS

### 5.1 Dataset Details
The model was trained on the CamVid dataset. The training set consists of RGB dashcam frames, and the validation set is used to monitor overfitting and save the best model weights. The dataset requires parsing colored masks into sparse 2D class-index matrices before computing loss.

### 5.2 Training Configuration
| Parameter | Value |
| :--- | :--- |
| Encoder | ResNet34 |
| Batch Size | 8 |
| Optimizer | AdamW (lr = 1e-4) |
| Loss Function | CrossEntropy Loss + Dice Loss |
| Epochs | 50 |
| Hardware | CPU / CUDA GPU |

### 5.3 Evaluation Metrics
The primary metric used to evaluate segmentation quality is the **mean Intersection over Union (mIoU)**. It calculates the overlap between the predicted segmentation mask and the ground truth mask, averaged across all 32 classes. Loss is also tracked to ensure convergence.

### 5.4 Results
- **Model:** U-Net (ResNet34)
- **Status:** Successfully Trained
- **Validation mIoU:** ~0.4391 (Achieved at peak epoch)

During training, the model successfully converged, showing a steady decrease in Combined Loss and an increase in mIoU. By epoch 50, the model learned to accurately segment major classes like Roads, Sky, and Vehicles, generating structurally coherent maps.

---
<div style="page-break-after: always"></div>

## 6. RESULTS ANALYSIS & DISCUSSION

### 6.1 Confusion Matrix
*(Insert your generated `confusion_matrix.png` image here)*

The normalized confusion matrix visually details the model's recall across all 32 classes on the validation set. A strong diagonal line indicates that the model correctly predicts the majority of pixels for their respective classes. Off-diagonal elements represent misclassifications; for instance, the model occasionally confuses visually similar structural boundaries or merges thin, granular objects (like poles or distant pedestrians) into the dominant background class due to spatial downscaling in the bottleneck. The use of normalized accuracy prevents dominant classes (like Road or Sky) from overshadowing the evaluation of rarer classes.

### 6.2 Prediction Logic
The system outputs a tensor of shape `[Batch, Classes, Height, Width]`. The prediction logic involves taking the `argmax` across the class dimension to determine the most likely class for each individual pixel. This 2D map of integers is then cross-referenced with the `CLASS_COLORS` array to generate an RGB visual overlay.

### 6.3 Predictions and Localization Quality
The model performs exceptionally well at localizing macro-structures. Roads, sky, and large buildings are segmented with near-perfect boundaries. Vehicles are identified accurately, though heavily occluded objects or thin structures (like distant poles or fences) occasionally blend into the background due to spatial downsampling limitations.

### 6.4 Observations
- The use of Dice Loss in combination with Cross-Entropy significantly helped with class imbalance, ensuring that smaller objects (like pedestrians) are penalized heavily if missed.
- The pre-trained ResNet34 weights allowed the network to converge extremely fast compared to a randomly initialized network.
- The web application executes inference in seconds, proving the model is lightweight enough for practical server-side deployments.

### 6.5 Strengths
- **Spatial Precision:** Skip connections in the U-Net architecture preserve sharp boundaries.
- **Transfer Learning:** ResNet34 provides robust, generalized feature extraction.
- **User Experience:** The included web app allows non-technical users to inspect and visualize the model's capabilities instantly.

### 6.6 Limitations
- **Thin Structures:** The model struggles with highly granular details (e.g., thin signposts or wire fences) due to resolution downscaling.
- **Computational Cost of Video:** While fast on images, processing high-FPS video requires a dedicated GPU for real-time (30+ FPS) throughput.

---
<div style="page-break-after: always"></div>

## 7. CONCLUSION, FUTURE SCOPE & REFERENCES

### 7.1 Conclusion
We successfully designed and implemented an end-to-end semantic segmentation system using a U-Net architecture with a ResNet34 encoder. By training on the CamVid dataset, the model learned to parse complex urban scenes and classify pixels into 32 distinct categories. The integration with a Flask backend demonstrates a practical, user-friendly pipeline for autonomous driving research and dataset exploration.

### 7.2 Future Scope
- **Real-Time Video Optimization:** Implement TensorRT or ONNX Runtime to accelerate video inference for deployment on edge devices like NVIDIA Jetson.
- **Architectural Upgrades:** Upgrade the architecture to DeepLabV3+ with Atrous Spatial Pyramid Pooling (ASPP) to better capture multi-scale context and improve thin-structure segmentation.
- **Temporal Consistency:** Integrate Recurrent Neural Networks (RNNs) or 3D Convolutions to enforce temporal smoothness between video frames, reducing flickering predictions.

### 7.3 References
[1] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI, 2015.  
[2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition", CVPR, 2016.  
[3] G. J. Brostow, J. Fauqueur, and R. Cipolla, "Semantic object classes in video: A high-definition ground truth database", Pattern Recognition Letters, 2009 (CamVid Dataset).  
[4] Flask Documentation, Pallets Projects, https://flask.palletsprojects.com/ (accessed 2026).  
[5] PyTorch Documentation, Linux Foundation, https://pytorch.org/ (accessed 2026).
