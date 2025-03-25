<div align="center">
  <img src="ReadmeImages/S1.png" alt="Demo Image 1" align="center" width="400px"/>
  <img src="ReadmeImages/S2.png" alt="Demo Image 2" align="center" width="400px"/>
  <h1>Cartograph Automata</h1>
  <p>By Khoi, Rokawoo, Arya, Mehroj</p>
</div>



> [!CAUTION]
> :star: Our Codefest Project is really Based!

## Abstract

Cartograph Automata is a computer vision-based navigation system for mobile robots in a simulated environment - here we simulated warehouse environments. It combines object detection with simple path-planning logic to help robots avoid obstacles and move safely in real time.

We used up to 700 labeled images to train an SSD ResNet50 model that detects 3 main targeted objects: people, shelves, and paths. The training process was tracked using loss graphs, showing strong performance over 14,000 steps.

The system uses segmentation and bounding-box mapping to create a simple obstacle map from image data. If the robot's path crosses an obstacle, it calculates a safe detour, follows it, and then returns to the original path once the way is clear.

<b>Applications:</b> In warehouses, it helps mobile robots avoid workers or obstacles. In healthcare, delivery robots can use vision-based rerouting to safely navigate crowded hospital hallways. In retail, it allows shopping assistants or cleaning bots to move smoothly around people and displays in real time. The system also serves as a valuable learning tool in robotics education, offering hands-on experience with computer vision, object detection, and planning in a simulated environment. Beyond that, it provides a lightweight testbed for evaluating AI model performance in real-time navigation scenarios, making it useful for both research and practical deployment.

## Demo

[Demo Video](#)

## Project Media

<div align="center">
  <img src="/ReadmeImages/seq1_center_20250228-043837.png" alt="D1" width="400px"/>
  <img src="/ReadmeImages/seq37_center_20250228-042041.png" alt="D2" width="400px"/>
  <img src="/ReadmeImages/seq34_left_20250228-043507.png" alt="D3" width="400px"/>
  <img src="/ReadmeImages/seq2_center_20250228-035304.png" alt="D4" width="400px"/>
  <img src="/ReadmeImages/robot.png" alt="D3" width="400px"/>
  <img src="/ReadmeImages/robot1.png" alt="D4" width="400px"/>
</div>

## Obejct Detection

<div align="center">
  <img src="/ReadmeImages/object_detection2.png" alt="D1" width="400px"/>
  <img src="/ReadmeImages/object_detection3.png" alt="D1" width="400px"/>
  <img src="/ReadmeImages/object_detection4.png" alt="D1" width="400px"/>
  <img src="/ReadmeImages/object_detection5.png" alt="D1" width="400px"/>
</div>

## Image Segmentation
<div align="center">
  <img src="/ReadmeImages/ImageSegmentationOutput1.png" alt="D1" width="400px"/>
  <img src="/ReadmeImages/ImageSegmentationOutput2.png" alt="D1" width="400px"/>

</div>

# **Training Graphs**
We visualize various loss metrics during the training process to track model performance and convergence. Below are the key loss functions plotted:

| Loss Type               | Description |
|-------------------------|-------------|
| **Classification Loss** | Measures how well the model classifies objects, typically using Cross-Entropy Loss. |
| **Localization Loss**   | Penalizes incorrect bounding box predictions, often using Smooth L1 or IoU Loss. |
| **Regularization Loss** | Helps prevent overfitting by adding constraints (e.g., L2 weight decay). |
| **Total Loss**         | Sum of all losses, representing overall optimization progress. |
| **Learning Rate**       | Tracks how the learning rate changes over training steps. |

## **Learning - Loss Graph**
Below is a consolidated graph displaying all loss metrics over training steps:
<div align="center">
  <img src="/ReadmeImages/classification_loss.png" alt="D1" width="400px"/>
  <img src="/ReadmeImages/localization_loss.png" alt="D1" width="400px"/>
  <img src="/ReadmeImages/regularization_loss.png" alt="D1" width="400px"/>
  <img src="/ReadmeImages/learning_rate.png" alt="D1" width="400px"/>
  <img src="/ReadmeImages/total_loss.png" alt="D1" width="400px"/>
</div>
<div align="center">
  <img src="/ReadmeImages/complete_graph.png" alt="D1" width="600px"/>
</div>

## :mag: Key Features

- **Intelligent Path Deviation**: Makes contextual decisions about when to leave programmed paths
- **Obstacle Classification**: Distinguishes between static obstacles, humans, and other mobile objects
- **Return-to-Path Algorithm**: Efficiently returns to optimal routes after obstacle avoidance
- **Simulation Validated**: Tested in various Webots environments including warehouses and public spaces
- **Hybrid Sensing Integration**: Combines traditional sensors with computer vision for robust navigation

## 🛠️ Libraries and Tools Used

- **TensorFlow Model Zoo**: For fine-tuning the SSD ResNet50 V1 FPN 640x640 (RetinaNet50) model
- **Webots**: Professional robot simulator for testing and validation
- **Python**: Primary programming language
- **OpenCV**: For additional image processing
