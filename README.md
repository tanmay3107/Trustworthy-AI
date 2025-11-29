🩻 Trustworthy AI: Explainable Medical DiagnosticsOverviewTrustworthy
 
AI is a computer vision system designed to detect lung pathologies from Chest X-rays while providing visual explanations for its decisions.
 
In healthcare, a "black box" prediction (e.g., "99% Confidence") is insufficient and dangerous. This project bridges the trust gap by implementing Grad-CAM (Gradient-weighted Class Activation Mapping). It generates heatmaps that overlay the X-ray, showing doctors exactly where the model is looking inside the lungs to make its diagnosis.
 
✨ Key Features
 
Multi-Class Diagnosis: Successfully classifies 4 distinct conditions: Covid-19, Viral Pneumonia, Tuberculosis, and Normal.
 
Explainability Engine (XAI): Overlays "Attention Heatmaps" on X-rays to validate that predictions are based on relevant anatomical features (like lung opacities) rather than artifacts.
 
Deep Learning Core: Powered by a ResNet50 architecture, fine-tuned on a diverse dataset of medical images.
 
False Positive Reduction: The visualization layer helps identify "Data Leakage" (e.g., ensuring the model isn't cheating by reading text labels or hospital tags).
 
🛠️ Tech Stack
 
Deep Learning: PyTorch (Torchvision, Autocast Mixed Precision)Model Architecture: ResNet50 (Fine-tuned)Explainability: pytorch-grad-camHardware Acceleration: Optimized for NVIDIA GPUs (RTX Series) via CUDA.Visualization: OpenCV & Matplotlib
 
🚀 How to Run
 
1. Install Dependencies
    pip install torch torchvision opencv-python pytorch-grad-cam matplotlib

2. Train the ModelThe system includes a smart training script that auto-detects classes and uses Mixed Precision for faster training on RTX GPUs.
    python train_medical_model.py

Output: Saves medical_resnet_4class.pth3. Generate Diagnosis & HeatmapRun the XAI dashboard to process an image and see the "Why" behind the prediction.
    python medical_xai_4class.py

📸 Results: The "Trust" Factor

    Before Training,                                                After Training (Trustworthy)
(Heatmap focuses on random noise/text),                             (Heatmap focuses on Lung Opacity)
"<img src=""assets/untrained_heatmap.jpg"" width=""300"" alt=""Bad Heatmap"">","<img src=""assets/trained_heatmap.jpg"" width=""300"" alt=""Good Heatmap"">"