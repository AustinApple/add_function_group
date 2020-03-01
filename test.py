# import feature
# from substitution_multi import canonize_ls, canonize
from rdkit import Chem

#list_test = canonize_ls(['So1cccc1', 'Sc1ccco1', 'Sc1ccoc1', 'Sc1ccoc1', 'Sc1ccco1'])
# SO1C=CC=C1


def canonize(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True, canonical=True)

smi_canon = canonize('So1cccc1')

# print(smi_canon)
# Chem.MolFromSmiles('SO1C=CC=C1')




