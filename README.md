# AFToolkit

**AFToolkit** is a framework for molecular modeling of proteins with AlphaFold2 derived representations. 


## Installation
```angular2html
git clone https://github.com/AIRI-Institute/AFToolkit.git

cd AFToolkit

conda env create --name=aftoolkit_env -f environment.yml
source activate aftoolkit_env

pip install .
```

## Data
We prepared 3 models for protein stability task usinf SVM, MLP and Catboost adapters, and 1 model for prediction protein-protein affinity using SVM adapter.

You can download models using foolowing links:
* Models trained on protein stability task:
```angular2htm
mkdir model/monomer
mkdir model/monomer/pair+lddt_logits+plddt/
cd model/monomer/pair+lddt_logits+plddt/
wget -O https://bioinformatics-kardymon.obs.ru-moscow-1.hc.sbercloud.ru/AFToolkit/models/monomer/trained_catboost_concat_nomultitrain_aggmutpos_multisum.pkl
wget -O https://bioinformatics-kardymon.obs.ru-moscow-1.hc.sbercloud.ru/AFToolkit/models/monomer/trained_mlp_concat_nomultitrain_aggmutpos_multisum.pkl
wget -O https://bioinformatics-kardymon.obs.ru-moscow-1.hc.sbercloud.ru/AFToolkit/models/monomer/trained_svm_concat_nomultitrain_aggmutpos_multisum.pkl
```
* Model trained on protein-protein affinity task:
```angular2htm
mkdir model/multimer
mkdir model/multimer/pair+lddt_logits+plddt/
cd model/multimer/pair+lddt_logits+plddt/
wget -O https://bioinformatics-kardymon.obs.ru-moscow-1.hc.sbercloud.ru/AFToolkit/models//multimer/trained_svm_concat_nomultitrain_aggmutpos_multisum.pkl
```

To test our models, please, download task csv-files and features pkl-files. File with features for training monomer adapter models are too large, please, generate them using `run_protein_task` scripts and `cdna+PROSTATA_mut_idxs.csv` task file.

* Test data for protein stability task:
```angular2htm
cd data
#task files
wget -O https://bioinformatics-kardymon.obs.ru-moscow-1.hc.sbercloud.ru/AFToolkit/data/stability_task_files.zip
#s669 features pkl-files
wget -O https://bioinformatics-kardymon.obs.ru-moscow-1.hc.sbercloud.ru/AFToolkit/data/s699_pkls.zip
#ssym features pkl-files
wget -O https://bioinformatics-kardymon.obs.ru-moscow-1.hc.sbercloud.ru/AFToolkit/data/ssym_pkls.zip
#PTmul features pkl-files
wget -O https://bioinformatics-kardymon.obs.ru-moscow-1.hc.sbercloud.ru/AFToolkit/data/protherm_pkls.zip
#cDNA de novo features pkl-files
wget -O https://bioinformatics-kardymon.obs.ru-moscow-1.hc.sbercloud.ru/AFToolkit/data/denovo_pkls.zip
#cDNA indels features pkl-files
wget -O https://bioinformatics-kardymon.obs.ru-moscow-1.hc.sbercloud.ru/AFToolkit/data/cdna_indel_pkls.zip
```

* Test data for protein-protein affinity task:
```angular2htm
cd data
#task files
wget -O https://bioinformatics-kardymon.obs.ru-moscow-1.hc.sbercloud.ru/AFToolkit/data/ppi_task_files.zip
#S4196 features pkl-files
wget -O https://bioinformatics-kardymon.obs.ru-moscow-1.hc.sbercloud.ru/AFToolkit/data/S4169_pkl.zip
#C380 features pkl-files
wget -O https://bioinformatics-kardymon.obs.ru-moscow-1.hc.sbercloud.ru/AFToolkit/data/C380_pkl.zip
```

## Usage

### <a name="af_features"></a>Extracting AlphaFold Features
You can calculate AF2 features for certain protein or protein complex: 
```angular2html
run_protein_task \
    --pdb  data/examples/PDB/1A7V.pdb \
    --chain A \
    --mutations A:A66H \
    -n 3 \
    --return-all-cycles \
    -o ./output
```

```angular2html
run_protein_complex_task \
    --pdb  data/examples/PDB/1CSE.pdb \
    --chains-for-protein I \
    --mutations I:L38S \
    -n 1 \
    --return-all-cycles \
    -o ./output
```

You can calculate AF2 features for the dataset using task-file. Please, see examples.
```angular2html
run_protein_task \
    -i data/examples/monomer_task.csv \
    --output-dir ./output \
    -n 3
```
```angular2html
run_protein_complex_task \
    -i data/examples/ppi_task.csv \
    --output-dir ./output \
    -n 3
```

To see description for all optional arguments of run_protein_task and run_protein_complex_task scripts use help option.
```angular2html
run_protein_task -h
run_protein_complex_task -h
```

You can also using implemented functions inside your python code and considered protein embeddings for other tasks or save recycled protein structure in PDB file.
```angular2html
from AFToolKit.processing.protein_task import ProteinTask
from AFToolKit.processing.openfold_wrapper import OpenFoldWrapper
from AFToolKit.processing.arg_parser import parse_mutations
from AFToolKit.processing.utils import save_to_pdb


of_wrapper = OpenFoldWrapper(device='cuda:0',
                             inference_n_recycle=3,
                             always_use_template=False,
                             side_chain_mask=False,
                             return_all_cycles=False)
of_wrapper.init_model()

protein_task = ProteinTask()
protein_task.set_input_protein_task(protein_path='data/examples/PDB/1A0F.pdb',
                                    chains=['A'])
protein_task.set_task_mutants(parse_mutations('A:S11A'))
protein_task.set_observable_positions()
logger.info('Calculate embeddings')
protein_task.evaluate(of_wrapper=of_wrapper,
                      store_of_protein=args.store_protein
                      )
#save recycled wildtype structure
save_to_pdb(protein_task.get_wildtype_protein_of(), 'data/examples/test_save.pdb')
```

### Evaluation
You can evaluate our pretrained SVM, MLP and CatBoost models.

Please see examples for protein stability task in `run_test_monomer_example.py` file.
For affinity prediction use `run_test_multimer_example.py` file and for rosetta energy prediction use `run_test_perresidue_example.py`.


### Training
You can create your own dataset using `run_protein_task` and `run_protein_complex_task` scripts and train the corresponding model or create your own adapter.

Please see examples in `train_{adaptor}_monomer.py`,`train_svm_multimer.py` and `train_svm_perresidue` files.


## License
This code is provided under MIT License:

The MIT License (MIT) Copyright (c) 2016 AYLIEN Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
