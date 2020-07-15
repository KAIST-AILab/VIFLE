# VIFLE
Implementation for Variational Inference using Future Likelihood Estimates
##Requirements
To install requirements:

```sh
conda env create -f environment.yml
conda activate seq_vi
```

To run the code, download pianoroll dataset into 
datasets/pianoroll/(dataset)
(ex. datasets/pianoroll/jsb.pkl)

## Polyphonic Music Datasets
To run vifle on jsb datasets, run this command:

```sh
python seq_vi.py --pid=1
```

or, you can change algorithm and dataset using following command:

```sh
python seq_vi.py --algorithm=(algorithm_name) --dataset_name=(dataset_name)
```

(ex. python seq_vi.py --algorithm="vifle" --dataset_name="jsb")

## References

If this repository helps you in your academic research, you are encouraged to cite our paper. Here is an example bibtex:
```bibtex
@inproceedings{KimEtal.ICML20,
  author    = {Geon-Hyeong Kim and Youngsoo Jang and Hongseok Yang and Kee-Eung Kim},
  title     = {Variational Inference for Sequential Data with Future Likelihood Estimates},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year      = {2020}
}
