# [ICML 2026] MoLF: Mixture-of-Latent-Flow
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

## Quick Start
### 0.0 Data Preprocessing

- Download HEST1k from huggingface https://huggingface.co/datasets/MahmoodLab/hest. to `data/HEST1K` or [PATH/TO/YOUR/HEST1K]
- Please follow respective pathology foundation model (PFM) repositories for preprocessing. For example, preprocessed data with PFM [uni_v2](https://github.com/mahmoodlab/UNI) is in `data/sample_processed_HEST1K/uni` with names `[sample_id]_embeddings.pt` which contains barcodes and correspongding uni features. This path is referred in `utils/custom_dataset.py` for dataloading.
- Also download hest_bench https://huggingface.co/datasets/MahmoodLab/hest-bench and put it in `data/Hest_Bench` [PATH/TO/YOUR/hest_bench] for fixed held-out test set and HVG-50 genes list
- Data split is merged from hest-bench and provided in `data/Hest_Bench/hest_pathway_clean_split_0` and `data/Hest_Bench/hest_pathway_clean_split_1`.


Following code could be used to preprocecss HEST1k dataset.

```
python3 utils/pfm_preprocesing.py
```

### 1.0 Trianing
- Stage 1 gene VAE
```
python3 gene_autoencoder_train.py -f configs/training_config_latent_moe.yml
```

- stage 2 latent flow matching 
```
python3 train.py -f configs/training_config_latent_moe.yml
```

Pretrained weights can also be downloaded from [huggingface](https://huggingface.co/HuSusu/MoLF).

### 2.0 Sampling
```
# test set sampling
python3 sampling.py -l training_log/run_20260506_162107_molf -g 2.0

# external mouse validation sampling
python3 sampling.py -l training_log/run_20260506_162107_molf -g 2.0 -e
```
### 3.0 Post-hoc analysis & visualization
TODOs

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**. For full details, see the [LICENSE](LICENSE) file.


### 📝 Citation
If you find our work useful, please consider cite us. 
```bibtex
@inproceedings{hu2026molf,
  author    = {Hu, Susu and Speidel, Stefanie},
  title     = {MoLF: Mixture-of-Latent-Flow for Pan-Cancer Spatial Gene Expression Prediction from Histology},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
  url       = {https://icml.cc/virtual/2026/poster/63940}
}