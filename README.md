# HPLT3-PerRef

This repository contains HPLT3-PerRef source code.

The classifier used can be found on [Hugging Face](https://huggingface.co/alrazz-n/quality_classifier).

The HPLT3-PerRef dataset is available on [Hugging Face](https://huggingface.co/datasets/alrazz-n/HPLT3-PerRef).

`Ham_Spam` folder contains code for fine-tuning XLM-R and BGE-M3, finding the threshold, and creating plots.

`binary_dataset` contains the human-annotated data used for training the classifier, in Hugging Face dataset format.

`perplexity` is used for the Intrinsic Corpus-Quality Check.


# Usage

If you use this dataset in your research or projects, please cite the following paper:

# Classifiers performance

| Metric        | BGE-M3  | XLM-R   |
|---------------|---------|---------|
| Accuracy      | 96.91%  | 97.56%  |
| Macro F1      | 96.58%  | 97.31%  |
| Weighted F1   | 96.90%  | 97.56%  |
| Total errors  | 19/615  | 15/615  |
