"""
auther: Mathias Stokkebye Nissen
last updated: 18-dec-23
version 0.2
"""


import random
import numpy as np

import ase
from ase.neighborlist import NeighborList

from acat.build import add_adsorbate_to_site


def new_ad_tag(atoms):
    all_tags = []
    for atom in atoms:
        t = atom.tag
        all_tags.append(t)
    tags = list(set(all_tags))
    for i in range(len(tags)+1):
        if i in tags:
            continue
        else:
            return i
        
def add_random_adsorbats(surface, SAS_sites, ad_species, ad_site_height, add_num = 1):

    atoms = surface.copy()

    for _ in range(add_num):
        i = random.choice(np.arange(len(SAS_sites)))
        ad_specie = random.choice(ad_species)  
        add_adsorbate_to_site(atoms, adsorbate = ad_specie,
                               site = SAS_sites[i], height = ad_site_height[i])

        new_tag = new_ad_tag(atoms)
        for i in range(1, 1 + len(ad_specie)):
            atoms[-i].tag = new_tag

    return atoms, new_tag

def test_atoms_doubbel(atoms):
    all_pos = atoms.get_positions()
    all_pos = np.round(all_pos, 3)

    for i in range(len(atoms)):
        pos = all_pos[i,:]
        for j in range(i+1,len(atoms)):
            row = all_pos[j,:]
            if np.array_equal(row, pos):
                return True
    return False

def remove_to_closs(surface, ad_tag, dist = 1.5/2, print_result = False):
    atoms = surface.copy()
    del atoms[[atom.index for atom in atoms if atom.tag == 0]]

    remove_ad = False

    if test_atoms_doubbel(atoms):
        remove_ad = True

    nl = NeighborList(np.ones(len(atoms)) * dist, bothways = True, skin=0)
    nl.update(atoms)
    
    atoms_index = []
    for atom in atoms:
        if atom.tag == ad_tag:
            atoms_index.append( atom.index )
            
    for index in atoms_index:
        for neigbor_index in nl.get_neighbors(index)[0]:
            if neigbor_index not in atoms_index:
                remove_ad = True

    atoms = surface.copy()
    if remove_ad:
        del atoms[[atom.index for atom in atoms if atom.tag == ad_tag]]

    if print_result:
        print('Removed adsorbat:', remove_ad)

    return atoms, remove_ad

def generate_random_adsorbate_layer(surface, SAS_sites, num_surfaces, add_num, ad_species,
                                     ad_site_height, too_close_dist = 1.5/2, add_persistence = 20):

    out_surfaces = []

    X = []

    for i in range(num_surfaces):
        atoms = surface.copy()

        persistence = add_persistence
        j = 0
        while j < add_num:
            atoms, new_tag = add_random_adsorbats(atoms, SAS_sites, ad_species, ad_site_height)

            atoms, removed_ad = remove_to_closs(atoms, new_tag, dist = too_close_dist)
            if removed_ad:
                persistence -= 1
                if persistence < 0:
                    j += 1
            else:
                j += 1
            
            X.append(atoms)

        out_surfaces.append(atoms)

    return out_surfaces

def adsorbate_decomposed_test(system, tag, max_d):
    """
    Tests if a single adsorbat has decomposed.
    system: ASE atoms object of the entire syrface-adsorbats system
    tag: the tag of the adsorbat which should be tested
    max_d: max distance which the adsorbate atoms may be apart
    
    retruns: False if not decomposed and True otherewise
    """

    atoms = remove_atoms_not_of_tag(system, tag) # removes all atoms exept adsorbate
    if len(atoms) == 1:
        return False

    max_d = max_d/2 # NeighborList views atoms as overlapping if their radii overlap
    nl = NeighborList(np.ones(len(atoms)) * max_d, bothways = True, self_interaction = False, skin=0)
    nl.update(atoms)

    for atom in atoms:
        i = atom.index
        if len(nl.get_neighbors(i)[0]) == 0: # If any of the adsorbate is within max_d
            return True
    return False

def adsorbates_decomposed_test(system, max_d):
    """
    Wrapper around adsorbate_decomposed_test, for testing all
    adsorbats of the system.
    system: ASE atoms object of the entire syrface-adsorbats system
    max_d: max distance which the adsorbate atoms may be apart
    
    retruns: False if not decomposed and True otherewise
    """

    atoms = system.copy()

    tags = atoms.get_tags() # Get all tags in system (atoms object)
    tags = list(set(tags)) # Remove all non-unique
    tags.remove(0) # Remove the tag of the surface

    tags_out = []
    for tag in tags:
        if adsorbate_decomposed_test(atoms, tag, max_d):
            tags_out.append(tag)

    if len(tags_out) == 0:
        return False
    else:
        return tags_out

def adsorbates_too_close_test(system, min_d):
    """
    system: ASE atoms object of the entire syrface-adsorbats system.
    min_d: minimum distance (seperation) adsorbats should have.
    
    retruns: False if not non are closer than min_d or a list of with
    the adsorbate tags which are with in min_d of an other adsorbat.
    """

    atoms = system.copy()
    atoms = remove_atoms_of_tag(atoms, 0) # Removing surface/slab atoms, given as copy 

    min_d = min_d/2 # NeighborList views atoms as overlapping if their radii overlap
    nl = NeighborList(np.ones(len(atoms)) * min_d, bothways = True, self_interaction = False, skin=0)
    nl.update(atoms)

    too_close_tags = []
    for i in range(len(atoms)):
        
        if atoms[i].tag in too_close_tags:
            continue

        neigbor_indices = nl.get_neighbors(i)[0]
        for index in neigbor_indices:
            if atoms[i].tag != atoms[index].tag:
                too_close_tags = too_close_tags + [atoms[i].tag, atoms[index].tag]
    
    too_close_tags = list(set(too_close_tags)) # Removing duplicats:

    if len(too_close_tags) == 0:
        return False
    else:
        return too_close_tags

def too_far_from_surface_test(system, max_d):
    """
    system: ASE atoms object of the entire syrface-adsorbats system.
    max_d: the maximum distance which an adsorbat may have form the
    surface slab.

    returns: False if no adsorbats are futhere away than max_d or a
    list of with the tags of the adsorbats which are.
    """
    atoms = system.copy()
    tags = atoms.get_tags() # Get all tags in system (atoms object)
    tags = list(set(tags)) # Remove all non-unique
    tags.remove(0) # Remove surface tag: 0
    too_far = tags.copy() # All tags starts as marked as too far away
    # then thay are remove from this list if thay are close to the surface.

    max_d = max_d/2 # NeighborList views atoms as overlapping if their radii overlap
    nl = NeighborList(np.ones(len(atoms)) * max_d, bothways = True, self_interaction = False, skin=0)
    nl.update(atoms)

    for atom in system:
        if atom.tag not in too_far:
            continue

        neigbor_indices = nl.get_neighbors(atom.index)[0]
        for index in neigbor_indices:
            if atoms[index].tag == 0:
                too_far.remove(atom.tag)
                break

    if len(too_far) == 0:
        return False
    else:
        return too_far
    

def remove_atoms_of_tag(atoms, tag):
    atoms = atoms.copy()
    del atoms[[atom.index for atom in atoms if atom.tag == tag]]
    return atoms

def remove_atoms_not_of_tag(atoms, tag):
    atoms = atoms.copy()

    tags = atoms.get_tags() # Get all tags in system (atoms object)
    tags = list(set(tags)) # Remove all non-unique
    tags.remove(tag) # Remove tag, which is to be kept

    for t in tags:
        atoms = remove_atoms_of_tag(atoms, t)
    return atoms
