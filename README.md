# add-functional-group
This code is to add functional groups on molecules. 
## Environment setting 
### via Anaconda
```
conda env create -f environment.yml
source activate molecule_feature_predict
```

## Tutorial 
### Preparing the materials
Prepare a `.csv` file including molecules SMILES on which you want to add functional groups (default=mainmol.csv). And also prepare a `.csv` file including functional groups' name and SMARTS (default=func.csv). 
### Adding functional groups on molecules
`python substitution.py`

There are some arguments.

```
  -h, --help            show this help message and exit
  -i MAINMOL, --mainmol MAINMOL
                        main molecule
  -f FUNCTION, --function FUNCTION
                        functional group you want to add on molecules
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        the output path
  -n NUMBER, --number NUMBER
                        how many functional groups you want to add
  --multi               add different kinds of functional groups
  --singel              add one kind of function groups
```

For example, adding four one kind of functional groups on molecules.
`python substitution.py -n 4 --single`
adding three different kinds of functional groups on molecules.
`python substitution.py -n 4 --single`
