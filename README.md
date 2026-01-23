# Interactive Language-Queryable Gaussian Scenes

**Developed by:** Sergejs Zahovskis, Dmitry Knorre, James Conant, and Yael Fassbind  
**Course:** 2025 Mixed Reality at ETH Zurich  
**Supervised by:** Alexander Veicht

## 🌟 Overview

This project enables users to interact with photorealistic 3D scenes in VR using natural language voice commands (e.g., "highlight the bonsai tree"). While standard VR applications rely on cumbersome controllers for text input, our system allows for intuitive, immersive scene manipulation directly on a Meta Quest 3.

We overcome the hardware limitations of mobile VR (8GB RAM, Adreno 740 GPU) by offloading heavy language embedding computations to a local server while maintaining real-time rendering natively on the headset.

This is a fork of [UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting).

### Key Features

- **Real-Time Rendering**: Native 3D Gaussian Splatting on VR headset at stable frame rates (~12 FPS)
- **Semantic Understanding**: Integration of Occam's Language Gaussian Splatting (LGS) to link 3D objects with natural language
- **Voice-Driven Interaction**: Speech-to-text via OpenAI's Whisper and semantic querying via CLIP (ViT-B-16)
- **Multiple Rendering Modes**: 
  - Standard colored Gaussian Splats
  - Occam Similarities (grayscale with heatmap highlighting)
  - Occam Colored Similarities (heatmap overlay on colored scene)

## 🛠️ Requirements & Setup

### Prerequisites

- **Unity Version:** `2023.1.14f1`
- **Hardware:** Meta Quest 3 and a laptop/PC (both must be on the same Wi-Fi network)
- **Python:** 3.x with virtual environment support
- **Operating System:** The project was developed on macOS/Linux (the server uses `source venv/bin/activate`)

### 1. Repository & Assets

1. Clone this repository:
```bash
   git clone [repository-url]
   cd [project-directory]
```

2. Download the required `.ply` files:
   - [occam_bonsai_mcmc.ply - https://polybox.ethz.ch/index.php/s/Qb9QYnDNMqS8XjC]
   - [occam_meeting_room_mcmc.ply - https://polybox.ethz.ch/index.php/s/3t9sLdEF6nKATAw]

3. Place both `.ply` files inside the `occam_backend/` directory of the project.

### 2. Unity Configuration

1. Open the project in **Unity 2023.1.14f1**
2. Go to **File > Build Settings**, select **Android**, and click **Switch Platform**
3. In the Project window, navigate to the `Scenes` folder and load the main scene
4. In the Hierarchy, you will see two game objects: `meeting_room` and `bonsai`
   - **Important:** Enable only ONE of these objects at a time (disable the other)
   - The enabled object determines which scene you will view
5. Connect your Meta Quest 3 via USB
6. Click **Build and Run** to deploy the application to the headset

## 🚀 Running the Application

The system requires both the VR app and the Python backend server running simultaneously.

**Critical:** The server and VR application must be configured for the **same scene** (either bonsai or meeting_room).

### 1. Launch the Backend Server

1. Open a terminal and navigate to the `occam_backend/` directory:
```bash
   cd occam_backend/
```

2. Activate the virtual environment:
```bash
   source venv/bin/activate
```

3. Launch the server for your chosen scene:
   - **For Bonsai scene:**
```bash
     python occam_server.py occam_bonsai_mcmc.ply
```
   - **For Meeting Room scene:**
```bash
     python occam_server.py occam_meeting_room_mcmc.ply
```

### 2. Use the VR Application

1. Put on the Meta Quest 3 headset
2. Ensure the headset is connected to the same Wi-Fi network as your PC
3. Launch the deployed application on the headset
4. Use the controllers to interact:
   - **Left Trigger:** Start/end voice input
   - **Right Trigger (toggle):** Switch between rendering modes (gaussian splats / occam similarities (black and white))
   - **A Button (right controller):** Activate Occam Colored Similarities mode. Press **A** again or **right trigger** to switch to other rendering modes
5. Press **Left trigger**, say the name of an object (e.g., "bonsai tree", "camera", "push toy") and **press the trigger again** 
6. The system will process your query and highlight matching objects in real-time

### Rendering Modes

The current rendering mode is displayed in a panel at the top-left corner of your view:
1. **Standard Mode:** Colored Gaussian Splats
2. **Occam Similarities:** Grayscale scene with heatmap highlighting
3. **Occam Colored Similarities:** Original colors with relevancy heatmap overlay

## 🎨 Creating Custom Scenes

The meeting room scene was created using our customized version of SplatFactory. If you want to create your own language-queryable Gaussian scenes:

1. Visit our [SplatFactory fork](https://github.com/Mitdia/SplatFactory) for scene capture and training
2. Follow the pipeline described in the report (Section 3.3):
   - Capture the scene (video or multiple overlapping photos)
   - Prepare a COLMAP dataset with camera poses
   - Train the 3D Gaussian scene
   - Extend with language features using Occam's LGS approach
3. Optionally apply MCMC pruning to reduce Gaussian count for better performance
4. Export the final scene as a `.ply` file with language feature fields

The bonsai scene used in this project is from the [NeRF baselines paper](https://github.com/jkulhanek/nerfbaselines).


## 📂 Project Structure
```
├── Assets/                    # Unity application components
│   └── ...                   # Interaction scripts, UI, camera controls
├── occam_backend/            # Python server
│   ├── occam_server.py      # Main server script
│   ├── venv/                # Virtual environment
│   └── *.ply                # Scene files (download separately)
└── package/                  # Custom Gaussian Splatting rendering package
    └── GaussianSplatRenderer.cs
```

## 🔬 Technical Implementation

### Rendering Innovation

Instead of alpha-blending full 512-dimensional language feature vectors (infeasible on mobile GPU), we:
1. Precompute cosine similarities between the user query and each Gaussian's language feature on the server
2. Send only scalar relevancy scores to the headset
3. Alpha-blend single scalar values during rendering, achieving performance comparable to standard RGBA rendering

### Relevancy Scoring

We use canonical queries ("object", "thing", "texture", "material") with softmax normalization to reduce noise and ensure highlighted items truly match user intent.

### Optimizations

- **MCMC Pruning:** Reduces Gaussian count for better frame rates
- **Near-Plane Culling:** Prevents flickering when camera is close to objects
- **Multi-Pass Rendering:** Optional colored overlay mode for enhanced visualization

## 📊 Evaluation Results

Our user study with 12 participants achieved:
- **SUS Score:** 78/100 (good usability)
- **UEQ-S Pragmatic Quality:** 1.542/3
- **UEQ-S Hedonic Quality:** 1.583/3
- **Voice Input Preference:** 72.7% of users preferred voice over keyboard

## ⚠️ Known Limitations

- Performance is limited by Gaussian count (~1 million max for stable frame rate)
- Requires external server connection (adds latency)
- Multi-pass rendering mode (Occam Colored Similarities) reduces frame rate and **may introduce intense flickering if used with meeting room or larger scene** 
- Object highlighting can be noisy in some cases

## 📄 References

This project builds upon:
- **3D Gaussian Splatting** ([Kerbl et al., 2023](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/))
- **Occam's LGS** ([Cheng et al., 2025](https://arxiv.org/abs/2411.14027))
- **LangSplat** ([Qin et al., 2023](https://arxiv.org/abs/2312.16084))
- **UnityGaussianSplatting** ([Aras Pranckevičius](https://github.com/aras-p/UnityGaussianSplatting))
- **SplatFactory** ([Our customized fork](https://github.com/Mitdia/SplatFactory))

## 🤝 Acknowledgments

Special thanks to Alexander Veicht for supervision and providing the Occam implementation.