## Lacune Segmentation

- Jihwan Min

<br>

<p align="center">
  <img width="900" alt="image" src="https://github.com/user-attachments/assets/2703a825-b148-499b-a509-a5ca2df2527c">
</p>

<p align="center">
  Figure 1. The overall structure of our proposed model
</p>

<br>
<br>

### Convolutional Block Attention Module (CBAM)

- CBAM module contains two main modules: Channel Attention module and Spatial Attention module.
- The channel attention module and the spatial attention module generate an attention map (which contains what and where to look) for each channel and space.
- The channel attention module identifies 'what' is the most meaningful part in the feature, while the spatial attention module focuses on 'where' the important information is.

<p align="center">
  <img width="400" alt="image" src="https://github.com/user-attachments/assets/b9c9c17a-afc2-4903-8db9-ad349c70c087">
</p>

<p align="center">
  Figure 2. Convolutional Block Attention Module
</p>

<br>

### Model (RLK-UNet with CBAM module

- RLK-UNet, basically UNet structure, contains Multi-scale Highlighting foregrounds modules which is essential to detect small lesion areas.
- ighlighting foreground regions at multiple scales, which allowed the network to better differentiate between Lacunes and non-Lacunes backgrounds, at multi-scale decoder layers
- To highlight critical information from low-level encoder features, we apply the CBAM module to the output function of the encoder to pass it on to the skip connection and concatenate it in the decoder with attention mechanism.
- The loss function used is a combination of Dice loss and weighted BCE loss. Since Lacunes are very small, class imbalance occurs, and to address this, the combination of Dice and foreground-weighted BCE loss can make the model focus on small Lacunes areas and minimize False Negative components.
