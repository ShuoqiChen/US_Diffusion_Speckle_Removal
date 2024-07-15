*Candidate Diffusion models:*

DDIM (Denoising Diffusion Implicit Models) - Song et al. "Denoising Diffusion Implicit Models" https://arxiv.org/abs/2010.02502

PNDM (Pseudo Numerical Methods for Diffusion Models) - Liu et al. "Pseudo Numerical Methods for Diffusion Models on Manifolds" https://arxiv.org/abs/2202.09778

DDPM (Denoising Diffusion Probabilistic Models) - Ho et al. "Denoising Diffusion Probabilistic Models" https://arxiv.org/abs/2006.11239


The methods above of are all types of diffusion models used for generative tasks for high-quality, however, they do have different types of implementation complexities and other trade-offs

<!-- A table of comparison below: -->

<!-- Feature	DDPM	DDIM	PNDM
Type	Probabilistic	Deterministic	Hybrid (Numerical Methods)
Sampling Process	Probabilistic (many steps)	Deterministic (fewer steps)	Numerical approximation (fewer steps)
Inference Speed	Slow	Faster	Faster
Complexity	Moderate	High	High
Training Objective	Predict noise	Predict noise	Predict noise with numerical methods
Typical Architecture	U-Net	U-Net	U-Net or similar
Quality	High	High	High -->

To stop before generating complete noise in diffusion process

Noise Schedule Adjustment: Define a noise schedule that stops before complete noise, retaining more structure in the image.

Forward Diffusion Process: Modify the forward process to add noise up to the defined level.

Reverse Diffusion Process: Perform the reverse diffusion using the same modified noise schedule to generate images.


Other methods to constrain the forward noise generation process

Regularization: Apply regularization techniques during noise addition to constrain the noisy image.

Guidance: Use a pre-trained classifier or feature extractor to guide the noise addition process.

Style Constraints: Incorporate style features using style transfer techniques to ensure the noisy image retains certain stylistic elements.