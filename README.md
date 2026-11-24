# Modelos Generativos en Deep Learning: DCGAN para Rostros Humanos y Fine-Tuning de Stable Diffusion con `diffusers`

**Autor:** Oscar  
**Programa:** Programa Académico en Inteligencia  
**Actividad:** Generación — Implementación Práctica de GANs y Modelos de Difusión Latente  
**Fecha:** Noviembre de 2026  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## Descripción del Proyecto

Este repositorio contiene la implementación práctica, el cuaderno de Google Colab y el código fuente modular para el estudio comparativo de dos arquitecturas generativas de *Deep Learning* aplicadas a la síntesis de imágenes:

1. **DCGAN (*Deep Convolutional Generative Adversarial Network*):**
   * Basada en el artículo *Implementing a Generative Adversarial Network (GAN/DCGAN) to Draw Human Faces* y en los trabajos seminales de Goodfellow et al. (2014) y Radford et al. (2015).
   * Implementa en **PyTorch** las redes **Generator ($G$)** (`ConvTranspose2d`, `BatchNorm2d`, `ReLU`, `Tanh`) y **Discriminator ($D$)** (`Conv2d`, `BatchNorm2d`, `LeakyReLU(0.2)`, `Sigmoid`) sobre imágenes RGB de rostros humanos ($64 \times 64$) del conjunto *Labeled Faces in the Wild (LFW)*.
2. **Latent Diffusion Models & Fine-Tuning de Stable Diffusion v1.5 con `diffusers`:**
   * Basado en el artículo *Latent Diffusion Models: The Architecture behind Stable Diffusion* (Rombach et al., 2022) y adaptación de bajo rango **LoRA** (Hu et al., 2022).
   * Utiliza la librería **`diffusers` de Hugging Face** (`AutoencoderKL`, `UNet2DConditionModel`, `CLIPTextModel`, `DDPMScheduler`, `StableDiffusionPipeline`) junto con `peft` para ejecutar *fine-tuning* eficiente en el espacio latente perceptual ($4 \times 64 \times 64$) sobre GPU Tesla T4 en Google Colab.

---

## Estructura del Repositorio

```text
proyecto_modelos_generativos/
├── DCGAN_Stable_Diffusion_Generacion_Colab.ipynb  # Notebook principal para Google Colab
├── Informe_Modelos_Generativos_GAN_Stable_Diffusion_APA.docx  # Documento técnico en formato APA 7
├── README.md                                      # Documentación técnica del repositorio
├── requirements.txt                               # Dependencias de Python y Deep Learning
├── figures/                                       # Evidencias gráficas y resultados de entrenamiento
│   ├── figura1_arquitectura_gan_ldm.png
│   ├── figura2_curvas_entrenamiento_dcgan.png
│   ├── figura3_progresion_rostros_dcgan.png
│   ├── figura4_finetuning_stable_diffusion.png
│   └── figura5_comparativa_desempeno_vram.png
└── src/                                           # Módulos de código fuente en Python
    ├── dcgan_faces.py                             # Implementación modular de Generator y Discriminator
    └── stable_diffusion_finetune.py               # Script de Fine-Tuning LoRA con Hugging Face diffusers
```

---

## Ejecución en Google Colab

1. Abrir el cuaderno **`DCGAN_Stable_Diffusion_Generacion_Colab.ipynb`** en Google Colaboratory.
2. En el menú superior, seleccionar **Entorno de ejecución $\rightarrow$ Cambiar tipo de entorno de ejecución $\rightarrow$ Acelerador por hardware: GPU (T4)**.
3. Ejecutar todas las celdas secuencialmente (**Entorno de ejecución $\rightarrow$ Ejecutar todas**).
