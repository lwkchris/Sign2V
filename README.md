I'm thrilled to share that I've been awarded the Silver Medal at the IEIG2025!
<img src="https://imgur.com/It3JMsf.png" alt="Booth">
<img src="https://imgur.com/eeHHGF5" alt="Award">
---

# Project: Sign2V

This project is designed to recognize and process American Sign Language (ASL). 
It includes tools for preprocessing data, training machine learning models, testing accuracy, and deploying an 
application for real-time sign language recognition and speech-to-text conversion to achieve 2-Ways Conversation.

<img src="https://imgur.com/It3JMsf.png" alt="Logo" width="250">

---

# Project Directory Structure

```plaintext
project/
├── Application/                
│   ├── App.py                          # Main application for real-time ASL recognition
│   ├── bubble_style.py                 # Handles UI/UX elements (text block bubbles)
│   ├── input_processing.py             # Processes user input 
│   ├── landmark_display.py             # Displays hands' landmarks
│   └── speech_to_text_processing.py    # Converts speech-to-text 
│
├── Labeling/                   # Tools for labeling and flipping gesture data
│   ├── DataCollection.py           # Script for collecting and labeling ASL
│   └── Flipping.py                 
│
├── Preprocessing/              # Prepares ASL datasets for training
│   ├── asl_dataset/                # Original dataset 
│   ├── augmented_asl_dataset/      # Augmented dataset (Same structure as asl_dataset directory)
│   ├── preprocessing_data.npz      # Preprocessed dataset stored as a `.npz` file
│   ├── DataAugmentation.py         # Augmentation techniques for improving dataset quality
│   └── preprocessing.py            # Scripts for preprocessing raw ASL images
│
├── Resource/                   # Placeholder for assets (e.g., images, icons, etc.)
│
├── TestTools/                  # Quick tests for different modules
│   ├── Camera_test.py              # Tests camera functionality
│   ├── Model_quick_test.py         # Verifies model predictions
│   └── Speech_to_Text_test.py      # Tests speech-to-text functionality
│
├── Training/                   # Training scripts and saved models
│   ├── model_dir/                  # Directory to store trained models
│   │   ├── .pth                
│   │   └── .joblib             
│   ├── model.py                    # Model architecture
│   └── training.py                 # Script for training models
│
├── Validation/                 # Validation different setting of models
│   ├── model_dir/                  # Directory for validated models
│   │   ├── .pth                    
│   │   └── .joblib                 
│   ├── model.py                    # Model used during validation
│   └── training.py                 # Script for validating/training models
│
├── README.md  
└── requirements.txt     
```

---

# Getting started

---

## Current Configuration
 - **Python 3.10** or above
 - **CUDA 12.6** for GPU acceleration. 
 - Pytorch: https://pytorch.org/get-started/locally/ 
   - **Command:** ```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126```
 - Libraries: Install necessary Python packages listed in requirements.txt.
   - **Command:** ```pip install -r requirements.txt```

<br>

---

## 1. Data Preparation & Preprocessing
1. Prepare the dataset
   - Quick-start dataset: https://www.kaggle.com/datasets/ayuraj/asl-dataset or
   - Use ```DataCollecting/DataCollection.py``` script to capture and label the raw image data through your webcam (Optional)

<img src="https://imgur.com/9MieL1D.png" alt="Labeling_Demo" width="400" height="300">

2. Place your raw ASL data in the ```Preprocessing/asl_dataset/``` directory.
   #### Dataset Structure
   ```plaintext
   asl_dataset/
   ├── a/
   │   ├── .jpeg
   │   ├── .jpeg
   │   └── ...
   ├── ...
   ├── z/
   ├── Hello/
   ├── ...
   ├── Yes/
   └── No/
   ```
3. Run the ```Preprocessing/DataAugmentation.py``` script to augment into: ```Preprocessing/Augmented_asl_dataset/```
4. Run the ```Preprocessing/preprocessing.py``` script to extract the coordinates of hands' node and save as ```Preprocessing/preprocessed_data.npz/```
   - The process may take a long time to complete.
   - Please ignore the warnings of MediaPipe feedback requests.
     - For example, ```W0000 00:00:1746050869.710636   13528 inference_feedback_manager.cc:114] Feedback manager requires a model...```
   - Undetectable data, if encountered, may be logged or printed during execution.

<br>

---

## 2. Model Training
1. Run ```Training/Training.py``` script to start training the model
2. Choose the model you would like to apply (**CNN**, **FNN**, **MLP**, or all at once)
3. Wait for the training process
4. Confirmed that the chosen model & encoder has been saved in ```Training/model_dir/``` directory.

<img src="https://imgur.com/G0ZLJ2c.png" alt="Model_Performance" width="1000">

<br>

---

## 3. Application

Run ```Application/App.py``` script to execute the app.

<img src="https://imgur.com/sbLO2ln.png" alt="App_Demo" width="500">

<br>

---

## Other

```Validating\training.py```: Optimize the epoch setting

From ```TestTools\``` directory,
- ```Camera_test.py```: Test camera accessibility
- ```GPU_test.py```: Optimize the batch size setting
- ```Model_quick_test.py```: Verify the deployment of the trained model.
- ```Speech_to_Text.py```: Test the speech-to-text functionality (requires an internet connection)
