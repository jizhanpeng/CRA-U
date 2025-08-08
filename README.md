# CRA-U: Lightweight U-Net with Component Ranking Attention for Skin Lesion Segmentation

<hr />

> **Abstract:** *Melanoma is a highly malignant skin disease, for which early and accurate detection of lesions is crucial, as this significantly enhances the effectiveness of treatment. Compared with conventional manual examination methods, computer-aided diagnostic techniques based on automatic image segmentation have demonstrated significant application potential due to their high reproducibility and low-cost characteristics. Transformer is widely adopted in mainstream image segmentation methods due to its superior global image modeling capabilities. However, the self-attention in Transformers suffers from \emph{$O(n^2)$} time complexity, creating a bottleneck for real-time skin lesion segmentation. In this paper, we propose a new framework, termed \textbf{CRA-U}, which aims to accelerate the segmentation of skin lesions. First, images are inputted into two preprocessing stages to reduce the impact of interference factors on lesion region segmentation. Subsequently, the preprocessed images are fed into a modified U-Net for key region segmentation. In this process, this paper proposes to achieve global feature fusion through the \textbf{Component Ranking Attention (CR-Attention)}. This attention mechanism deeply integrates the low computational complexity of linear attention with a non-linear reweighting mechanism. Through the \textbf{Ranking Function}, CR-Attention effectively mitigates the deficiency of conventional linear attention in focusing capability. We evaluate our method on four public skin lesion datasets, demonstrating performance advantages over state-of-the-art methods. In addition, we evaluate the generalizability of our method on two typical medical image segmentation tasks. Compared to TransUNet, our method achieves superior performance while requiring only 1/41 of its parameter count.* 

<hr />

###  Dataset Download and Preperation

The datasets used in the paper can be downloaded from the following locations:

[ISIC(2016-2018)](https://challenge.isic-archive.com/data/#2018)

[PH2](https://www.fc.up.pt/addi/ph2%20database.html)

[2018 DSB](https://www.kaggle.com/c/data-science-bowl-2018)

[BUSI](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)


The training data should be placed in ``` inputs/``` directory .

### Training 

```
python train.py
```





Contact me at jizhanpeng@sues.edu.cn if there is any problem.



