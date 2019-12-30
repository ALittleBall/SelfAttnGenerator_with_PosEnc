# SelfAttnGenerator_with_PosEnc

Based on EdgeConnect paper. The Generator only use 
convolution layers in the generator. Though 
convolution can increase limited the receptive field, 
each pixel cannot have a global view of all the image.

This code adds self attention layers in the generator
network. Also the position encoding and multihead
attention are added in the self attention layers.

The reason why I add position encoding in the attention 
is that I think relative position of each pixel is also 
important, to help concentrate on adjacent pixels. Or 
the pixels may pay much less important attention to the 
irrelevant area. And this may cause GAN to generate 
the inpainting area negatively affected by the pixels
far away from the mask.


Reference:

@misc{zhang2018selfattention,
    title={Self-Attention Generative Adversarial Networks},
    author={Han Zhang and Ian Goodfellow and Dimitris Metaxas and Augustus Odena},
    year={2018},
    eprint={1805.08318},
    archivePrefix={arXiv},
    primaryClass={stat.ML}
}
