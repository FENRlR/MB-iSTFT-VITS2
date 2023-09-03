# MB-iSTFT-VITS2

![Alt text](resources/image6.png)

A... [vits2_pytorch](https://github.com/p0p4k/vits2_pytorch) and [MB-iSTFT-VITS](https://github.com/MasayaKawamura/MB-iSTFT-VITS) hybrid... Gods, an abomination! Who created this atrocity?

This is an experimental build. Does not guarantee performance, therefore. 

According to [shigabeev](https://github.com/shigabeev)'s [experiment](https://github.com/FENRlR/MB-iSTFT-VITS2/issues/2) it can now dare claim the word SOTA (at least for Russian).
 

## pre-requisites
1. Python >= 3.8
2. CUDA
3. [Pytorch](https://pytorch.org/get-started/previous-versions/#v1131) version 1.13.1 (+cu117)
4. Clone this repository
5. Install python requirements. Please refer [requirements.txt](requirements.txt)
    ~~1. You may need to install espeak first: `apt-get install espeak`~~
6. Prepare datasets
    1. ex) Download and extract the LJ Speech dataset, then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs DUMMY1`
7. Build Monotonic Alignment Search and run preprocessing if you use your own datasets.

```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
```
## Setting json file in [configs](configs)

| Model | How to set up json file in [configs](configs) | Sample of json file configuration|
| :---: | :---: | :---: |
| iSTFT-VITS2 | ```"istft_vits": true, ```<br>``` "upsample_rates": [8,8], ``` | istft_vits2_base.json |
| MB-iSTFT-VITS2 | ```"subbands": 4,```<br>```"mb_istft_vits": true, ```<br>``` "upsample_rates": [4,4], ``` | mb_istft_vits2_base.json |
| MS-iSTFT-VITS2 | ```"subbands": 4,```<br>```"ms_istft_vits": true, ```<br>``` "upsample_rates": [4,4], ``` | ms_istft_vits2_base.json |
| Mini-iSTFT-VITS2 | ```"istft_vits": true, ```<br>``` "upsample_rates": [8,8], ```<br>```"hidden_channels": 96, ```<br>```"n_layers": 3,``` | mini_istft_vits2_base.json |
| Mini-MB-iSTFT-VITS2 | ```"subbands": 4,```<br>```"mb_istft_vits": true, ```<br>``` "upsample_rates": [4,4], ```<br>```"hidden_channels": 96, ```<br>```"n_layers": 3,```<br>```"upsample_initial_channel": 256,``` | mini_mb_istft_vits2_base.json |

## Training Example
```sh
python train.py -c configs/mini_mb_istft_vits2_base.json -m models/test
```

## Credits
- [jaywalnut310/vits](https://github.com/jaywalnut310/vits)
- [p0p4k/vits2_pytorch](https://github.com/p0p4k/vits2_pytorch)
- [MasayaKawamura/MB-iSTFT-VITS](https://github.com/MasayaKawamura/MB-iSTFT-VITS)
- [ORI-Muchim/PolyLangVITS](https://github.com/ORI-Muchim/PolyLangVITS)
- [tonnetonne814/MB-iSTFT-VITS-44100-Ja](https://github.com/tonnetonne814/MB-iSTFT-VITS-44100-Ja)
