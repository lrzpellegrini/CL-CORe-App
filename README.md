# The CL CORe App

The CORe (**C**ontinual **O**bject **Re**cognition) App is an *Artificial Vision* application for Android devices that can classify objects from the [CORe50 dataset](https://vlomonaco.github.io/core50/) and also **learn how to classify objects of new categories** by using the **AR1\* Continual Learning Algorithm** with **Latent Replay** (more info in the [Papers](#papers) sections).

All the inference and training tasks are performed on-device, offline and without accelerators. There are no particular hardware requirements:  1GB+ RAM and 100 MB of free storage space are recommended.

## News, download and demos

- \>\> Check the **[short video demo](https://youtu.be/Bs3tSjwbHa4)** on YouTube <<

- **Source code** published!

- **Paper** accepted at ESANN 2021! See the [Papers](#papers) section.

- You can **download** the app from the [release](https://github.com/lrzpellegrini/CL-CORe-App/releases) page.

- Vincenzo Lomonaco's post [*Latent Replay for Real-Time Continual Learing at the Edge*](https://medium.com/continual-ai/latent-replay-for-real-time-continual-learing-at-the-edge-9a083c899856) on Medium (**recommended**)!

## Installation
You can **download** the app from the release [release](https://github.com/lrzpellegrini/CL-CORe-App/releases) page.

To install the app, first enable the "unknown apps/origin" from the system settings. A quick guide can be found [here](https://www.androidauthority.com/how-to-install-apks-31494/). Then, transfer the *.apk* file to the device using a method of your choice. Install the application using a file manager.

## Usage
The app will first ask for permissions to access the camera and write to the phone storage. The camera access is required for obvious reasons while the external storage is needed to save the app state (which, in this way, can be exported to PC). The camera state will be stored in a folder named "CaffeAndroid".

1. The application starts in **inference mode**. The app comes with a model pretrained on the 10 [CORe50](https://vlomonaco.github.io/core50/) categories whose thumbnails will be shown in the lower part of the screen. A visual feedback about the classification score will appear around the thumbnails (the greener, the more the confidence for that category). A numerical feedback is also given for the top-3 categories. During this phase the app acts as a simple classifier. You can take an object from the already known categories and put it in front of the rear camera to get the confidence score. Beware that this app works only for **classification tasks** (not detection). Keep the object inside the **central square** (the greyed-out area is ignored).
2. The incremental **training phase** is triggered by **either tapping on a empty slot or on a existing category**. By tapping on an empty slot, a new category will be added (you will be asked to write its name). By tapping on an existing category, that category will be tuned to increase its accuracy on the new object (usually an object from the same semantic category...). Depending on the choice, few prompts will appear. Make your choices and proceed to the...
3. ... **countdown**. Three seconds countdown. The countdown will appear on the greyed-out overlay.
4. After the countdown, the app will start **gathering 100 images**. A progress bar will appear in the upper part of the screen showing the amount of images collected so far. During this phase it is recommended to **move the object around**, rotate it, change the background a little bit, and so on. Always keep the object in the central square!
5. The **training phase** will automatically start just after the gathering phase. It should take less than one second on most modern devices. At the end of the training phase, the new model will be stored in the "CaffeAndroid" folder and the app will switch back to inference mode.

You can also **reset the app** to its factory state by tapping on the icon in the upper-right part of the app. This will reset the current model, learnt categories, replay buffer, etcetera.

## Compiling
The full source code is now available! The app can be compiled using the latest version of Android Studio. However, consider *not* upgrading the Android build gradle plugin (and even Gradle itself) when prompted to do so.

Please note that the app does not execute Python code at all. A C++ version of the **AR1\*-free with Latent Replay** algorithm is used instead. If you are looking for the Python implementation, the Python source code for that algorithm is now available [here](https://github.com/lrzpellegrini/Latent-Replay)! An more modern implementation of the AR1\* algorithm is now packaged in the [Avalanche](https://avalanche.continualai.org) library, too.

## Papers
The paper **"Continual Learning at the Edge: Real-Time Training on Smartphone Devices"** has been accepted at ESANN 2021 and can be found [here](https://www.esann.org/sites/default/files/proceedings/2021/ES2021-136.pdf). The paper is available [on arXiv](https://arxiv.org/abs/2105.13127), too. The paper describes the application and shows the performance of the *AR1\*+Latent Replay* algorithm on a complex benchmark made of 225 incremental experiences.

    @inproceedings{pellegrini2021continual,
        author={Pellegrini, Lorenzo and Lomonaco, Vincenzo and Graffieti, Gabriele and Maltoni, Davide},
        title={Continual Learning at the Edge: Real-Time Training on Smartphone Devices},
        booktitle={29th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, {ESANN} 2021, Bruges, Belgium, October 6-8, 2021},
        pages={23--28},
        year={2021},
        url={https://www.esann.org/sites/default/files/proceedings/2021/ES2021-136.pdf},
        doi={10.14428/esann/2021.ES2021-136}
    }

However, if you need more details on the *Latent Replay* technique, please check our work **"Latent Replay for Real-Time Continual Learning"**, which has been accepted at IROS 2020. The paper is available [here](https://ieeexplore.ieee.org/document/9341460) along with its [source code](https://github.com/lrzpellegrini/Latent-Replay). The paper can also be found [on arXiv](https://arxiv.org/abs/1912.01100).

    @inproceedings{pellegrini2020latent,
        author={Pellegrini, Lorenzo and Graffieti, Gabriele and Lomonaco, Vincenzo and Maltoni, Davide},
        booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
        title={Latent Replay for Real-Time Continual Learning}, 
        year={2020},
        volume={},
        number={},
        pages={10203-10209},
        doi={10.1109/IROS45743.2020.9341460}
    }

## Previous works
You may also want to check our previous work, **"Rehearsal-Free Continual Learning over Small Non-I.I.D. Batches"**, which has been accepted at the 1st CLVision Workshop @ CVPR2020 ([paper here](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w15/Lomonaco_Rehearsal-Free_Continual_Learning_Over_Small_Non-I.I.D._Batches_CVPRW_2020_paper.pdf)). Its Python code is available [here](https://github.com/lrzpellegrini/Fine-Grained-Continual-Learning) and it includes a Caffe implementation of *AR1\** and *CWR\** algorithms without Latent Replay. The paper describes the **NICv2 benchmark** in more detail. The NICv2 benchmark generation procedure was used to generate the benchmark used to test the algorithm employed in the application.
