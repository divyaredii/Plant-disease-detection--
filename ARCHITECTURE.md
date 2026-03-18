# System Architecture

## Plant Disease Detection System

The system uses a multi-model ensemble approach to achieve high accuracy in plant disease detection. It combines three state-of-the-art deep learning models: ResNet50, EfficientNet-B3, and DenseNet121.

### 1. Detailed System Flowchart

```mermaid
graph TD
    %% Nodes
    User((User))
    UI[Web Interface / Flask App]
    
    subgraph Preprocessing
        ImgInput[Input Image]
        Resize[Resize 224x224]
        Norm[Normalization]
        Tensor[Convert to Tensor]
    end

    subgraph EnsembleModel [Ensemble Model System]
        direction TB
        M1[ResNet50<br/>Acc: 97.5%]
        M2[EfficientNet-B3<br/>Acc: 98.2%]
        M3[DenseNet121<br/>Acc: 96.8%]
        
        W1(Weight: 0.35)
        W2(Weight: 0.38)
        W3(Weight: 0.27)
        
        Avg{Weighted<br/>Average}
    end

    subgraph Output
        FinalPred[Final Prediction]
        Conf[Confidence Score]
        Details[Disease Info &<br/>Treatment]
    end

    %% Edge Connections
    User -->|Uploads| UI
    UI --> ImgInput
    ImgInput --> Resize --> Norm --> Tensor
    
    Tensor --> M1 & M2 & M3
    M1 -->|Softmax| W1
    M2 -->|Softmax| W2
    M3 -->|Softmax| W3
    
    W1 & W2 & W3 --> Avg
    
    Avg --> FinalPred
    FinalPred --> Conf
    FinalPred --> Details
    
    Details -->|Display| UI

    %% Styling
    classDef input fill:#f9f,stroke:#333,stroke-width:2px;
    classDef process fill:#e1f5fe,stroke:#0277bd,stroke-width:2px;
    classDef model fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,rx:10,ry:10;
    classDef result fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    
    class User,ImgInput,UI input;
    class Resize,Norm,Tensor,W1,W2,W3,Avg process;
    class M1,M2,M3 model;
    class FinalPred,Conf,Details result;
```

### 2. Execution Sequence Diagram

```mermaid
sequenceDiagram
    actor User
    participant Browser
    participant Server as Flask Server
    participant Model as Ensemble Engine
    
    User->>Browser: Uploads Plant Leaf Image
    Browser->>Server: POST /predict (Image Data)
    
    rect rgb(240, 248, 255)
        note right of Server: Preprocessing
        Server->>Server: Resize to 224x224
        Server->>Server: Normalize Pixel Values
    end
    
    Server->>Model: Forward Pass (Input Tensor)
    
    par Parallel Execution
        Model->>Model: ResNet50 Inference
    and
        Model->>Model: EfficientNet-B3 Inference
    and
        Model->>Model: DenseNet121 Inference
    end
    
    Model->>Model: Apply Weights (0.35, 0.38, 0.27)
    Model-->>Server: Final Class Probabilities
    
    Server->>Server: Identify Top Prediction
    Server-->>Browser: JSON Response (Disease, Accuracy, Remedies)
    
    Browser-->>User: Display Results & Treatment
```

### 3. Vision Transformer (ViT) Pipeline

The system now also supports a **Vision Transformer (ViT)** model — the state-of-the-art architecture for image classification tasks.

#### How ViT Works:
1. **Patch Embedding**: The input image (224×224) is split into 16×16 patches (196 patches total)
2. **Positional Encoding**: Each patch is embedded and given a positional encoding
3. **Transformer Encoder**: 12 layers of multi-head self-attention + FFN process the patches
4. **Classification Head**: The [CLS] token output is passed through a custom head for 39-class prediction

#### Key ViT Features:
- **Pretrained backbone**: ViT-Base-Patch16-224 pretrained on ImageNet-21K
- **Progressive unfreezing**: Head → last 2 blocks → full model
- **Label smoothing**: 0.1 for better generalization
- **Cosine warmup LR schedule**: Prevents early overfitting
- **Mixed precision training**: FP16 for 2× faster training on GPU

### 4. Key Components

| Component | Technology | Role |
|-----------|------------|------|
| **Frontend** | HTML5, CSS3, JS | User interface for image upload and result visualization |
| **Backend** | Python, Flask | API server, routing, and business logic |
| **Preprocessing** | OpenCV/PIL, NumPy | Image resizing, normalization, and tensor conversion |
| **Model 1** | Custom CNN (PyTorch) | Basic convolutional network baseline |
| **Model 2** | ResNet50 (PyTorch) | Deep residual learning with transfer learning |
| **Model 3** | **Vision Transformer (ViT)** | **State-of-the-art transformer-based image classification** |
| **Model 4** | EfficientNet-B3 | Optimized for efficiency and accuracy balance |
| **Model 5** | DenseNet121 | Feature reuse via dense connections |
| **Ensemble** | Custom Logic | Weighted averaging of model outputs |
| **ViT Library** | timm / HuggingFace transformers | Pretrained ViT models and utilities |
