# add-functional-group
This package is to add different functional groups on a molecule, and then this new molecule's electronic affinity (EA) and ionization energy (IE) can be predicted by machine learning model which has been trained by Material Project database.

## Environment setting 
### via Anaconda
```
conda env create -f environment.yml
source activate molecule_feature_predict
```

## Tutorial 
### Preparing the materials
Prepare a `.csv` file including molecules SMILES on which you want to add functional groups. And also prepare a `.csv` file including functional groups' name and SMARTS. 
### Adding functional groups on moleculs
`python substitution_multi.py`

There are some arguments.

```
  -i MAINMOL, --mainmol MAINMOL
                        main molecule file.csv
  -f FUNCTION, --function FUNCTION
                          functional groups.csv
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                                the output path
  -n NUMBER, --number NUMBER
                      how many functional groups you want to add
```

For example, I want to add four functional groups on a molecule.
`python substitution_multi.py -n 4`
