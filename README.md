# IPA Tokenizer

## Installation and Setup

To download the mapping file and  processed GLUE data, run the following command in your terminal:
For pre-processed GLUE data
```bash
gdown --folder https://drive.google.com/drive/folders/1K2nzP_PD8639ffX3GtDwJGCdWGNOsEBc

```
For mapping word to ipa 
``` bash
gdown https://drive.google.com/uc?id=1anM1SfUqLnIZ2DhrGCvw9YIND5tVwZk_

```

To test the tokenizer

Example:
```bash
python main.py --config_file config.yaml 
```
For finding an optimize parameter 
```bash
python optimize.py --config_file config_optimize.yaml 
```
