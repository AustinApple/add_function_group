# add-functional-group
This code is to add functional groups on molecules. 

The steps are as follows. 1. Label each atom of given molecule and the functional group. 2. Form the pair between one of the atoms from given molecule and one of the attached sites from functional group. 3. Add a bond according to the pairs to attach functional group to the given molecule. 4. Check whether the molecular structure violates chemical rules. 5. Label the generated molecules again. 6. Form the pairs. 7. Add a bond according to the pairs to form a ring. 8. Check whether the molecular structure violates chemical rules.

The below schematic diagram shows the workflow of the molecular modification.  
![image](https://github.com/AustinApple/add_function_group/blob/master/molecule_modification_algorithm.001.png)
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

For example,  
adding four one kind of functional groups on molecules.  
`python substitution.py -n 4 --single`  
adding three different kinds of functional groups on molecules.  
`python substitution.py -n 3 --multi`
