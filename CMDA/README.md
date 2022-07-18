# Confidence-guided Multi-level Domain Adaptation on Semantic Segmentation

### Data preparation

Download [Cityscapes](https://www.cityscapes-dataset.com/), [CycleGAN transferred GTA5](https://drive.google.com/open?id=1OBvYVz2ND4ipdfnkhSaseT8yu2ru5n5l) and [gta5 labels](https://drive.google.com/file/d/11E42F_4InoZTnoATi-Ob1yEHfz7lfZWg/view?usp=sharing).

### Generating Pseudo-labels and Confidence

Download a [pre-trained model](https://pan.baidu.com/s/1duLE04oKxoKAXp5Y5jjSiQ) (passward c4m7) and put it to the 'snapshots' dir.
Then run the code

```bash
python gen_pl_plc_sp.py --cfg config/gen_pl_plc_sp.yml --best_iter 0000
```

### Train

The GPU memory should be large enough to fit batch size 4. Then run the code

```bash
python train.py --cfg config/train_GTA2Cityscapes.yml
```

### Evaluation

Specify the paths of the checkpoint, and run the multi-scale test scripts `eval_mst.py`

```bash
python eval_mst.py
```

## Acknowledgements

This project is based on the following paper. We thank their
authors a lot for sharing the source code.

* [PDA](https://orca.cardiff.ac.uk/id/eprint/143144/1/PixIntraDA_MM_no_copyright.pdf)
