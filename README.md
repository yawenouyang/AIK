# AIK

This repository (under construction) is the implementation of [Towards Multi-label Unknown Intent Detection](https://yawenouyang.github.io/about/files/coling2022.pdf), which is accepted by COLING 2022.


## Requirements
Install requirements:
```bash
pip install -r requirements.txt
```

Other preparations:
- Download [bert-base-uncased](https://huggingface.co/bert-base-uncased/tree/main) and update the `base.yaml` file in the `configs` folder to locate it.  


## Usage

```bash
bash aik.sh
```

## Citation
```bibtex
@inproceedings{ouyang-etal-2022-towards,
    title = "Towards Multi-label Unknown Intent Detection",
    author = "Ouyang, Yawen  and
      Wu, Zhen  and
      Dai, Xinyu  and
      Huang, Shujian  and
      Chen, Jiajun",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.52",
    pages = "626--635",
}
```