

NUCLEOTIDE_MAPPING = {
    'DA': ('A', 0),
    'DT': ('T', 1),
    'DG': ('G', 2),
    'DC': ('C', 3)
}

AMINO_ACID_MAPPING = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

CONTACT_DISTANCE_CUTOFF = 4.5
NEIGHBOR_DISTANCE_CUTOFF = 6.0

MIN_DNA_LENGTH = 8
DNA_RATIO_THRESHOLD = 0.5

ATOM_RADIUS_DICT = {
    'C': 1.7,
    'N': 1.55,
    'O': 1.52,
    'S': 1.8,
}

BASE_ATOMS = [
    'N1', 'C2', 'N3', 'C4', 'C5', 'C6',
    
    'N7', 'C8', 'N9',
    
    'N2',
    'N4',
    'N6',
    'O2',
    'O4',
    'O6',
    'C7',
]

INTERACTION_DISCOUNT = {
    ('HYDROPHOBIC', 'HYDROPHOBIC'): 0.85,
    ('H_DONOR', 'H_ACCEPTOR'): 0.82,
    ('H_ACCEPTOR', 'H_DONOR'): 0.82,
    ('POSITIVE', 'NEGATIVE'): 0.8,
    ('NEGATIVE', 'POSITIVE'): 0.8,
}

ATOM_CHARGE_DICT = {
    ('N', 'ALL'): -0.5,
    ('O', 'ALL'): -0.5,
    ('C', 'ALL'): 0.0,
    
    ('N', 'ARG'): 1.0,
    ('N', 'LYS'): 1.0,
    ('N', 'HIS'): 0.5,
    ('O', 'ASP'): -1.0,
    ('O', 'GLU'): -1.0,
    
    ('O', 'SER'): -0.25,
    ('O', 'THR'): -0.25,
    ('O', 'TYR'): -0.25,
    ('N', 'GLN'): -0.25,
    ('N', 'ASN'): -0.25,
    
    ('S', 'CYS'): -0.25,
    ('S', 'MET'): 0.0,
}

SURFACE_RESOLUTION = 0.6
POINTS_PER_ANGSTROM = 0.8
MIN_SURFACE_POINTS = 1000
MAX_SURFACE_POINTS = 20000

NODE_FEATURE_DIM = 11
EDGE_FEATURE_DIM = 5
