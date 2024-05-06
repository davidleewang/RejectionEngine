# Overview:

Fine-tuning a large language model (Llama 3) to write "personalized" job rejection emails. Llama 3 was fine-tuned on ~700 job rejection emails collected from my inbox, which were received over the last few years.


# Running this code:

This project uses unsloth for faster inference and training of Llama 3. Some of the dependencies for unsloth are not available on Windows.

For running this code in Google Colab, run these commands in the Colab instance beforehand to download unsloth (and related modules):

```
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

!pip install --no-deps "xformers<0.0.26" trl peft accelerate bitsandbytes
```
