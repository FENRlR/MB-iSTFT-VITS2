# MB-iSTFT-VITS2

![Alt text](resources/image6.png)

A... [vits2_pytorch](https://github.com/p0p4k/vits2_pytorch) and [MB-iSTFT-VITS](https://github.com/MasayaKawamura/MB-iSTFT-VITS) hybrid... Gods, an abomination! Who created this atrocity?

This is an experimental build. Does not guarantee performance, therefore. 

According to [shigabeev](https://github.com/shigabeev)'s [experiment](https://github.com/FENRlR/MB-iSTFT-VITS2/issues/2), it can now dare claim the word SOTA for its performance (at least for Russian).
 

## pre-requisites
1. Python >= 3.8
2. CUDA
3. [Pytorch](https://pytorch.org/get-started/previous-versions/#v1131) version 1.13.1 (+cu117)
4. Clone this repository
5. Install python requirements.
   ```
   pip install -r requirements.txt
   ```
   If you want to use the Triton version of Super Monotonic Align, be sure to install triton by
   ```
   pip install triton
   ```
   and set ```"model": {"monotonic_align": "sma_triton"}``` in your configuration file.
   
   
    ~~1. You may need to install espeak first: `apt-get install espeak`~~
   
   If you want to proceed with those cleaned texts in [filelists](filelists), you need to install espeak.
   ```
   apt-get install espeak
   ```
7. Prepare datasets & configuration
   
    ~~1. ex) Download and extract the LJ Speech dataset, then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs DUMMY1`~~
   1. wav files (22050Hz Mono, PCM-16) 
   2. Prepare text files. One for training<sup>[(ex)](filelists/ljs_audio_text_train_filelist.txt)</sup> and one for validation<sup>[(ex)](filelists/ljs_audio_text_val_filelist.txt)</sup>. Split your dataset to each files. As shown in these examples, the datasets in validation file should be fewer than the training one, while being unique from those of training text.
      
      - Single speaker<sup>[(ex)](filelists/ljs_audio_text_test_filelist.txt)</sup>
      
      ```
      wavfile_path|transcript
      ```
      

      - Multi speaker<sup>[(ex)](filelists/vctk_audio_sid_text_test_filelist.txt)</sup>
      
      ```
      wavfile_path|speaker_id|transcript
      ```
   4. Run preprocessing with a [cleaner](text/cleaners.py) of your interest. You may change the [symbols](text/symbols.py) as well.
      - Single speaker
      ```
      python preprocess.py --text_index 1 --filelists PATH_TO_train.txt --text_cleaners CLEANER_NAME
      python preprocess.py --text_index 1 --filelists PATH_TO_val.txt --text_cleaners CLEANER_NAME
      ```
      
      - Multi speaker
      ```
      python preprocess.py --text_index 2 --filelists PATH_TO_train.txt --text_cleaners CLEANER_NAME
      python preprocess.py --text_index 2 --filelists PATH_TO_val.txt --text_cleaners CLEANER_NAME
      ```
      The resulting cleaned text would be like [this(single)](filelists/ljs_audio_text_test_filelist.txt.cleaned). <sup>[ex - multi](filelists/vctk_audio_sid_text_test_filelist.txt.cleaned)</sup> 
      
9. **(OPTIONAL)** Build Monotonic Alignment Search.
   
   This repo supports [supertone-inc](https://github.com/supertone-inc)'s [super-monotonic-align](https://github.com/supertone-inc/super-monotonic-align) for MAS. It removes Cython dependency(v1, v2, triton) and thus you do not have to build it anymore.
   However, you can still use the original version by the following 
   ```sh
   # Cython-version Monotonoic Alignment Search
   cd monotonic_align
   mkdir monotonic_align
   python setup.py build_ext --inplace
   ```
   and set ```"model": {"monotonic_align": "ma"}``` in your configuration file.
   
8. Edit [configurations](configs) based on files and cleaners you used.

## Setting json file in [configs](configs)
| Model | How to set up json file in [configs](configs) | Sample of json file configuration|
| :---: | :---: | :---: |
| iSTFT-VITS2 | ```"istft_vits": true, ```<br>``` "upsample_rates": [8,8], ``` | istft_vits2_base.json |
| MB-iSTFT-VITS2 | ```"subbands": 4,```<br>```"mb_istft_vits": true, ```<br>``` "upsample_rates": [4,4], ``` | mb_istft_vits2_base.json |
| MS-iSTFT-VITS2 | ```"subbands": 4,```<br>```"ms_istft_vits": true, ```<br>``` "upsample_rates": [4,4], ``` | ms_istft_vits2_base.json |
| Mini-iSTFT-VITS2 | ```"istft_vits": true, ```<br>``` "upsample_rates": [8,8], ```<br>```"hidden_channels": 96, ```<br>```"n_layers": 3,``` | mini_istft_vits2_base.json |
| Mini-MB-iSTFT-VITS2 | ```"subbands": 4,```<br>```"mb_istft_vits": true, ```<br>``` "upsample_rates": [4,4], ```<br>```"hidden_channels": 96, ```<br>```"n_layers": 3,```<br>```"upsample_initial_channel": 256,``` | mini_mb_istft_vits2_base.json |

### Monotonic Align configuration
 1. Super Monotonic Align (JIT_v1)
    ```
    "model": {"monotonic_align": "sma_v1"}
    ```
 2. Super Monotonic Align (JIT_v2)
    ```
    "model": {"monotonic_align": "sma_v2"}
    ```
 3. Super Monotonic Align (Triton)
    ```
    "model": {"monotonic_align": "sma_triton"}
    ```
 4. Monotonic Align (original Cython version)
    
    ```
    "model": {"monotonic_align": "ma"}
    ```
    (It will still work as the original version of monotonic align if not specified.)

## Training Example
```sh
# train_ms.py for multi speaker
python train.py -c configs/mb_istft_vits2_base.json -m models/test
```

## Credits
- [jaywalnut310/vits](https://github.com/jaywalnut310/vits)
- [p0p4k/vits2_pytorch](https://github.com/p0p4k/vits2_pytorch)
- [MasayaKawamura/MB-iSTFT-VITS](https://github.com/MasayaKawamura/MB-iSTFT-VITS)
- [ORI-Muchim/PolyLangVITS](https://github.com/ORI-Muchim/PolyLangVITS)
- [tonnetonne814/MB-iSTFT-VITS-44100-Ja](https://github.com/tonnetonne814/MB-iSTFT-VITS-44100-Ja)
- [misakiudon/MB-iSTFT-VITS-multilingual](https://github.com/misakiudon/MB-iSTFT-VITS-multilingual)
- [supertone-inc/super-monotonic-align](https://github.com/supertone-inc/super-monotonic-align)
