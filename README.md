# The CL CORe App

CL CORe App is an *Artificial Vision* application for Android devices that can classify objects from the CORe50 Dataset and also **learn how to classify objects of new classes** by using the **AR1\* Continual Learning Algorithm** with **Latent Replay** (more info in the [Source code](#source-code) and [Papers](#papers) sections).

All the inference and training tasks are made on-device, offline and without accelerators. The app actually supports ARM-v8 Android devices only, but we're planning on extending the support to ARM-v7 and Intel devices as well. There are no particular hardware requirements:  1GB+ RAM is recommended as well as 100 MB of free disk space.

You can download the app from the [release](https://github.com/lrzpellegrini/CL-CORe-App/releases) page.

# Source code
Full App source code coming soon!

In the meantime, the python source code for **AR1\*-free Latent Replay** (the algorithm used in this app) is now available: check it out [here](https://github.com/lrzpellegrini/Latent-Replay)! 

Note: the app doesn't esecute python code at all. Instead, a C++ version of that algorithm is used.

# Papers
Paper coming soon (just a few days)! It will describe the **Latent Replay** technique and will unveil the **app details**.


## Previous works
Our new paper is based on our previous work, *Fine-Grained Continual Learning*, which is available on arXiv: https://arxiv.org/abs/1907.03799.

Its python source code is available [here](https://github.com/lrzpellegrini/Fine-Grained-Continual-Learning) and it includes a Caffe implementation of **AR1\*** and **CWR\*** algorithms without Latent Replay.
