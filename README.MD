# MoLF: Mixture-of-Latent-Flow
### Pan-Cancer Spatial Gene Expression Prediction from Histology

[[📄 Paper](http://arxiv.org/abs/2602.02282)] [[🌐 Project Page](https://susuhu.github.io/MoLF/)] [[💻 Code](https://github.com/susuhu/MoLF)]

---

**MoLF** is a generative framework for scalable **pan-cancer histogenomics**. It utilizes a **Mixture-of-Experts (MoE)** velocity field within a **conditional Flow Matching** framework to predict spatial gene expression directly from standard H&E histology.

### 💡 Key Highlights
* **State-of-the-Art:** Outperforms existing methods on the HEST-1k pan-cancer benchmark.
* **Scalable Architecture:** Decouples diverse tissue patterns by routing morphological features through specialized experts.
* **Zero-Shot Generalization:** Demonstrates robust cross-species transfer (Human $\rightarrow$ Mouse) by learning invariant biological motifs.
* **Optimal Transport:** Efficiently maps random noise to a structured latent gene manifold in a single ODE step.

### 🛠️ Framework
1.  **Latent Compression:** A Transformer-VAE compresses high-dimensional gene expression into a structured manifold.
2.  **Generative Flow:** A Conditional Flow Matching model, parameterized by an MoE velocity field, generates expressions conditioned on H&E patches.

![MoLF Overview](figures/MoLF_overview.png)

### TODOS
upload code

### 📝 Citation
```bibtex
@article{hu2026molf,
  title={MoLF: Mixture-of-Latent-Flow for Pan-Cancer Spatial Gene Expression Prediction from Histology},
  author={Hu, Susu and Speidel, Stefanie},
  journal={arXiv preprint arXiv:2602.02282},
  year={2026}
}