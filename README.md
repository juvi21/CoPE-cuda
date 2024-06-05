# CoPE Contextual Position Encoding Kernels

I wrote some custom CUDA kernels for `flip`, which hasn't got an official PyTorch implementation (as I am aware of) , and `cumsum`, which has one but I implemented it also in a similar style for congruency. Both were created for the ROPE [paper](https://arxiv.org/abs/2405.18719). The CoPE core Implementation is taken from there.
**This repository is currently in toy status, so use it at your own risk.**

### Talk is cheap. How to run the code.

```sh

pip install -e .
```
After that, you should be able to import the custom Ops into your Python code.
Run Tests

```sh

pytest tests
```


Train MNIST

```sh

python train_mmnist.py
```

## Outlook: What to Do Next

- Introduce a benchmark in order to optimize the kernel better
- Make kernels faster
- Implement the entire forward pass in CUDA
- ~~Introduce `einops` just cause it's einops~~
- A very interesting idea was mentioned by https://www.youtube.com/@marinepower under https://www.youtube.com/watch?v=qcMsvU-wYZA by Gabriel Mongaras.
  > "Wonder if this method could be improved by having a new projection matrix of size [hidden_dim x 1] that computes the width of each token. We take the sigmoid, the cumulative sum, we do the interpolation as described, but we add it to the queries and keys, then do normal attention."
  - This requires a new (small) matrix but would allow us to use flash attention directly without needing a new CoPE kernel.
- Implement automatic testing
- Better README.md

Pull Requests are encouraged : )
