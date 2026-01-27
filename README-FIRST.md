<div align="center">

# Flux Inference Project
### 🎓 SYSU Computer Vision Coursework

<!-- 徽章区域 -->
[![SYSU](https://img.shields.io/badge/Institution-SYSU-005826?style=flat-square&logo=cplusplus)](https://www.sysu.edu.cn/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Weights-FFD21E?style=flat-square&logo=huggingface)](https://huggingface.co/masktrump19-sudo)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Demo-Gradio-FB7A24?style=flat-square&logo=gradio&logoColor=white)](https://gradio.app/)

<p align="center">
  <strong>Efficient inference implementation for the Flux architecture.</strong><br>
  Developed by <em>Huang Yongqing</em> for the CV course at Sun Yat-sen University.
</p>

</div>

---

## 📖 Project Overview

This repository contains the inference pipeline for the **[Flux](https://github.com/black-forest-labs/flux)** model. It provides optimized scripts to run image generation tasks efficiently using local environments.

> **✨ Key Feature:** The project supports both **Gradio** (Web UI) and **Streamlit** (Dashboard) for flexible demonstration.

## 📥 Model Weights (Hugging Face)

Due to GitHub's file size limits, the model weights (`.safetensors`) are hosted on my **Hugging Face** repository.

👉 **[Click here to Download Weights from Hugging Face](https://huggingface.co/masktrump19-sudo)** 
*(Please replace this link with your actual HF repo URL)*

### Setup Instructions
After downloading, please organize the files in the `checkpoints/` directory as follows:

```text
checkpoints/
├── flux1-dev.safetensors   <-- Main Model
└── ae.safetensors          <-- AutoEncoder
```

## 🚀 Getting Started

### 1. Environment Setup
Clone the repository and install dependencies:

```bash
# It is recommended to use conda or venv
pip install -r requirements.txt
```

### 2. Run Inference
Choose your preferred interface style:

| Interface Type | Command | Description |
| :--- | :--- | :--- |
| **Gradio** | `python demo_gr.py` | Interactive Web UI, suitable for quick testing. |
| **Streamlit** | `streamlit run demo_st.py` | Dashboard style, better for presentation. |

## 📂 Project Structure

```bash
.
├── src/                # 🧠 Core implementation logic
├── assets/             # 🎨 Images and static assets
├── checkpoints/        # ⚖️ Model weights (Download from HF)
├── output/             # 🖼️ Generated results (Auto-created)
├── demo_gr.py          # 🚀 Gradio startup script
├── demo_st.py          # 🚀 Streamlit startup script
└── requirements.txt    # 📦 Dependency list
```

## 👤 Author

<div align="left">
  <table>
    <tr>
      <td width="60px"><img src="https://github.com/masktrump19-sudo.png" width="50px" style="border-radius:50%"/></td>
      <td>
        <strong>Huang Yongqing</strong><br>
        Sun Yat-sen University (SYSU)<br>
        📧 <a href="mailto:huangyq296@mail2.sysu.edu.cn">huangyq296@mail2.sysu.edu.cn</a><br>
        🐙 <a href="https://github.com/masktrump19-sudo">@masktrump19-sudo</a>
      </td>
    </tr>
  </table>
</div>

## 📜 Disclaimer

This project is for **academic and educational purposes only**. 
The Flux model architecture and weights are properties of their respective owners (Black Forest Labs).