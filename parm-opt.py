import scipy
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import yaml
import math
from scipy.optimize import leastsq
import dask.distributed
from dask.distributed import Client,LocalCluster

method_dict = {'PM3':-7,'AM1':-2, 'RM1':-2, 'OM1':-5, 'OM2':-6, 'OM3':-8,'ODM2':-22, 'ODM3':-23,'XTB':-14}
            
at_num_dict = {'h':1, 'he':2, 'li':3, 'be':4, 'b':5,'c':6, 'n':7, 'o':8, 'f':9}
at_sym_dict = {1:"H", 2:"He", 3:"Li", 4:"Be", 5:"B", 6:"C", 7:"N",8:"O", 9:"F"}

def write_input (file, n_struc, n_atoms, at_nums, method_num, mol_num, charge):
    read_file = open(file[0],'r')
    if file[1] == 1:
        scale = 0.529
    else:
        scale = 1
        
    lines = read_file.readlines()
    read_file.close()
    write_file = open(f'mol{mol_num}.inp','w')
    
    n=0
    for structure in range(n_struc):
        if(structure == 0):
                    opt_file = open(f'opt{mol_num}.inp','w')
                    opt_file.write(f'iparok=1 nsav15=9 nprint=-4 kharge={charge} iform=1 iop={method_num} jop=0 igeom=1\n')
                    opt_file.write(f'Molecule {mol_num} Optimization\n\n')
        n +=1
        write_file.write(f'iparok=1 nsav15=9 nprint=-4 kharge={charge} iform=1 iop={method_num} jop=-1 igeom=1\n')
        write_file.write("Molecule #"+str(structure)+"\n\n")
        for atom in range(n_atoms):
            if(structure == 0):
                opt_file.write(f'{at_nums[atom]} ')
                [opt_file.write(f'{x*scale} 1 ') for x in np.array(lines[n].split(), dtype=float)]
                opt_file.write('\n')

            write_file.write(f'{at_nums[atom]} ')
            [write_file.write(f'{x*scale} 0 ') for x in np.array(lines[n].split(), dtype=float)]
            write_file.write('\n')
            n += 1
        write_file.write('0\n')
        if(structure == 0):
            opt_file.write('0\n')
            opt_file.close()
    write_file.close()

def read_input(file):
    with open(file) as f_in:
        lines = f_in.readlines() 
    n_parms= int(lines[0].split()[0])
    method = lines[1].split()[0]
    num_workers = 4
    if lines[1].split()[1].isdigit(): num_workers = int(lines[1].split()[1])
    method_num = method_dict[method.upper().replace("ORTHO", "")]
    n_molec = int(lines[2].split()[0])
    n_atoms=[]
    charge =[]
    structures=[]
    energy_files=[]
    structure_files = []
    n_geoms=[]
    n_weights=[]
    at_num=[[] for x in range(n_molec)]
    coords=[[] for x in range(n_molec)]
    geoms=[[] for x in range(n_molec)]
    weights=[[] for x in range(n_molec)]
    
    n = 3
    for mol in range(n_molec):
        n_atoms.append(int(lines[n].split()[0]))   # set the number of atoms in a molecule
        charge.append(int(lines[n+1].split()[0]))  # set the charge on the molecule
        structures.append(int(lines[n+2].split()[0])) # set the number of structures
        energy_files.append(lines[n+3].split()[0])  # set the file to find the energies
        structure_files.append([lines[n+4].split()[0],int(lines[n+4].split()[1])])# set the filename for the structures (the second flag is for atomic units = 1 or angstroms =0)
        if (structure_files[mol][1] == 1):
            scale = 0.529
        else:
            scale = 1.
        for atom in range(n_atoms[mol]):
            line = lines[n+5+atom].split()
            check = np.array(line[1:4],dtype=float)
            at_num[mol].append(at_num_dict[line[0].lower()])
            coords[mol].append(np.array([x*scale for x in check]))
        n_geoms.append(int(lines[n+5+n_atoms[mol]].split()[0])) #set the number of geometry calculations
        for geom in range(n_geoms[mol]):
            geoms[mol].append(lines[n+6+n_atoms[mol]+geom].split())
        n += n_atoms[mol]+n_geoms[mol]+6
        n_weights.append(int(lines[n].split()[0]))
        for weight in range(n_weights[mol]):
            weights[mol].append(lines[n+1+weight].split())
        n += n_weights[mol]+1
        
    return method_num, n_molec, n_atoms, charge, structures, energy_files, structure_files, at_num, coords, n_geoms, geoms, n_weights, weights, num_workers

def run_mndo(mol_num):
    os.system(f'mndo99 < mol{mol_num}.inp > mol{mol_num}.out')
    os.system(f'mv fort.15 mol{mol_num}.aux')
    os.system(f'mndo99 < opt{mol_num}.inp > opt{mol_num}.out')
    os.system(f'mv fort.15 opt{mol_num}.aux')
    
def read_opt(mol_num):
    intgeom = []
    optgeom = []
    with open(f'opt{mol_num}.out','r') as outfile:
        optlines = outfile.readlines()
    for n,line in enumerate(optlines):
        if "INPUT GEOMETRY" in line:
            for atoms in range(n_atoms[mol_num]):
                m = optlines[n+atoms+6].strip()
                o = m.split()[2::2]
                intgeom.append(o)   
        if "FINAL CARTESIAN GRADIENT NORM" in line:
            for atoms in range(n_atoms[mol_num]):
                z = optlines[n+atoms+8].strip()
                y = z.split()[2::2]
                optgeom.append(y)
    intgeom = np.array(intgeom).astype(float)
    optgeom = np.array(optgeom).astype(float)
    outfile.close()
    return intgeom, optgeom
    
def comp_geoms(n_molec):
    return_geoms = [] 
    for mol in range(n_molec):
        intgeom, optgeom = read_opt(mol)
        for geom in geoms[mol]:
            if geom[0] == 'bond':
                at_bond = ' '.join(map(str, geom))
                dist1 = np.linalg.norm(intgeom[int(geom[1])-1]-intgeom[int(geom[2])-1])
                dist4 = np.linalg.norm(optgeom[int(geom[1])-1]-optgeom[int(geom[2])-1])
                diff1 = dist1-dist4
                return_geoms.append(diff1)
                print(f'{at_bond} {dist1:.4} {dist4:.4} {diff1:.4}') #how many decimal places?
            if geom[0] == 'angle':    
                at_ang = ' '.join(map(str, geom))
                dot1 = np.dot((intgeom[int(geom[1])-1]-intgeom[int(geom[2])-1]), (intgeom[int(geom[3])-1]-intgeom[int(geom[2])-1]))
                dist2 = np.linalg.norm(intgeom[int(geom[1])-1]-intgeom[int(geom[2])-1])
                dist3 = np.linalg.norm(intgeom[int(geom[3])-1]-intgeom[int(geom[2])-1])
                cos1 = dot1/(dist2*dist3)
                angle1 = (math.acos(cos1))*57.295779513
                dot2 = np.dot((optgeom[int(geom[1])-1]-optgeom[int(geom[2])-1]), (optgeom[int(geom[3])-1]-optgeom[int(geom[2])-1]))
                dist5 = np.linalg.norm(optgeom[int(geom[1])-1]-optgeom[int(geom[2])-1])
                dist6 = np.linalg.norm(optgeom[int(geom[3])-1]-optgeom[int(geom[2])-1])
                cos2 = dot2/(dist5*dist6)
                angle2 = (math.acos(cos2))*57.295779513
                diff2 = angle1-angle2
                return_geoms.append(diff2)
                print(f'{at_ang} {angle1:.4} {angle2:.4} {diff2:.4}')
    return np.array(return_geoms) 

def xtb_geoms(n_molec):
    opt_geom = []
    for molecule in range(n_molec):
        file = open("opt_geom.dat","w")
        file.write(f"$coord\n")
        for atom in range(n_atoms[molecule]):
            file.write(f"{coords[molecule][atom][0]:17.14}{coords[molecule][atom][1]:17.14}{coords[molecule][atom][2]:17.14}      {at_sym_dict[at_num[molecule][atom]]}\n")
        file.write("$end\n")
        file.close()
        os.system("/share/apps/xtb-6.3.2/bin/xtb opt_geom.dat --opt normal")
        with open("xtbopt.dat","r") as opt:
            optLines = opt.readlines()
        for atom in range(n_atoms[molecule]):
            opt_geom.append(float(optLines[atom+1].split()[0])-float(coords[molecule][atom][0]))
            opt_geom.append(float(optLines[atom+1].split()[1])-float(coords[molecule][atom][1]))
            opt_geom.append(float(optLines[atom+1].split()[2])-float(coords[molecule][atom][2]))
    return opt_geom

def read_energies(n_molec):  
    energies=[]
    for mol in range(n_molec):
        energy = []
        with open(f'mol{mol}.out','r') as f:
            data = f.readlines()
        for n,line in enumerate(data):
            if "SCF TOTAL ENERGY" in line:
                energy.append(float(line.split()[3]))
        energies = np.hstack((energies,((np.array(energy)-np.min(energy))/27.2114)))
    return np.array(energies)

def read_abinito(energy_files):  
    energies=[]
    for file in energy_files:
        energy = []
        with open(file) as f:
            data = f.readlines()
        for line in data:
            energy.append(float(line.split()[0]))
        energies = np.hstack((energies,((np.array(energy)-np.min(energy)))))
    return np.array(energies)

def calc_fvec(structures, weights, n_geoms, geoms, method, n_atom, atom_num, energy):
    global spec_count
    if method == -14:
        energies = []
        new_energies = []
        fvec2 = []
        fvec = []
        for mol in range(n_molec):
            lazy_results = []
            files = xtb_method(n_atoms[mol],charge[mol],structures[mol],structure_files[mol][0],at_num[mol])
            for file in files:
                lazy_result = dask.delayed(xtb_run)(file)
                lazy_results.append(lazy_result)

            energies = np.hstack((energies,(dask.compute(*lazy_results))))# trigger computation in the background
            del lazy_results
        if True: #spec_count >= 100:
            for mol in range(n_molec):
                if mol == 0:
                    anp_int_spec(zero_energy(energies[:structures[mol]]),n_atoms[mol],atom_num[mol])
                else:
                    anp_int_spec(zero_energy(energies[structures[mol-1]:structures[mol-1]+structures[mol]]),n_atoms[mol],atom_num[mol])
                spectro_one = get_Spectro()
                if mol == 0:
                    anp_int_spec(read_abinito(energy_files)[:structures[mol]],n_atoms[mol],atom_num[mol])
                else:
                    anp_int_spec(read_abinito(energy_files[structures[mol-1]:structures[mol-1]+structures[mol]]),n_atoms[mol],atom_num[mol])
                spectro_two = get_Spectro()
                fvec2 = spectro_one-spectro_two
        energies = zero_energy(energies)
        fvec = np.hstack((fvec,hartree_to_wavenumber(energies-abinitio_energies))) #627.51*349.75
    
    else:
        fvec2 = []
        for mol in range(n_molec):
            run_mndo(mol)
        energies = read_energies(n_molec)
        fvec = hartree_to_wavenumber(energies-abinitio_energies)*627.51*349.75
        if spec_count >= 100:
            for mol in range(n_molec):
                if mol == 0:
                    anp_int_spec(ev_to_hartee(zero_energy(energies[:structures[mol]]),n_atoms[mol],atom_num[mol]))
                else:
                    anp_int_spec(ev_to_hartree(zero_energy(energies[structures[mol-1]:structures[mol-1]+structures[mol]]),n_atoms[mol],atom_num[mol]))
                spectro_one = getSpectro()
                if mol == 0:
                    anp_int_spec(read_abinito(energy_files)[:structures[mol]],n_atoms[mol],atom_num[mol])
                else:
                    anp_int_spec(read_abinito(energy_files)[structures[mol-1]:structures[mol-1]+structures[mol]],n_atoms[mol],atom_num[mol])
                spectro_two = get_Spectro()
                fvec2 = np.hstack((fvec2,(spectro_one-spectro_two)))
    w = np.ones(np.sum(structures+n_geoms)) 
    s = 0
    for mol in range(n_molec):
        for weight in weights[mol]:
            if weight[0] == '1':
                w[int(weight[3])-1+s:int(weight[4])-1+s] = w[int(weight[3])-1+s:int(weight[4])-1+s]*int(weight[1])
            if weight[0] == '2':
                for i in range(structures[mol]):
                    if str(i) in weight[2:]:
                        w[i]+=float(weight[1])
        s += structures[mol]
    for mol in range(n_molec):
        for geom in geoms[mol]:
            w[s] = w[s]*int(geom[-1])
            s += 1
    if method == -14:
        fvec = np.hstack((fvec,xtb_geoms(n_molec),fvec2))
    else:
        fvec = np.hstack((fvec,comp_geoms(n_molec),fvec2))
    print  ('rmsd  ' + str(np.sqrt(np.mean(np.square(fvec)))))
    fvec = fvec[:sum(structures)]*w
    spec_count+=1
    if energy: return fvec,energies
    else: 
        del energies
        return fvec

def read_parms(file):
    with open(file) as f_in:
            data = f_in.readlines() 
    parm_labels = []
    parm_vals   = []
    for line in data:
        if len(line.split()) == 0:
            break
        parm_labels.append(line.split()[0:2])
        parm_vals.append(float(line.split()[2]))
    return parm_labels, parm_vals

def write_parms(X,method):
    if method == -14:
        xtb_write_parms(X)
    else:
        with open('fort.14','w') as f:
            for i,line in enumerate(parm_labels):
                f.write(line[0]+'   '+line[1]+ ' ' +str(X[i])+'\n')

def big_loop(X,method):
    write_parms(X,method)  # Write the current set of parameters to fort.14
    fvec = calc_fvec(structures, weights, n_geoms, geoms, method, n_atoms, at_num, False) #,energies
    # if np.sum(n_geoms) > 0:
        # print  ('rmsd  ' + str(np.sqrt(np.mean(np.square(fvec[0:-np.sum(n_geoms)])))))
    #else:
       # print  ('rmsd  ' + str(np.sqrt(np.mean(np.square(fvec)))))
#      (f'RMSD {627.51*349.75*np.sqrt(np.mean(np.square(fvec)))}')   
    return fvec

def ev_to_hartree(energies):
    new_energies = [energy/27.2114 for energy in energies]
    return new_energies

def hartree_to_wavenumber(energies):
    new_energies = [energy *2.29371044869e17*6.62607015e-34*2.998e10 for energy in energies]
    return new_energies

def zero_energy(endData):
    energies = np.array(endData) 
    energies -= np.min(energies)
    return energies

def anp_int_spec(energies,n_atoms,at_num):
    
    energies = zero_energy(energies)
    iterData = iter(energies) 
    
    anpass_header    = "../anpass.header"
    anpass_footer    = "../anpass1.footer"
    intder_header1   = "../new-geom.in"
    intder_header2   = "../intder.in"
    spectro_template = "../spectro.in"
    dispDat          = "../disp.dat"
    fileName = "freq_files"
    
    if not os.path.isdir(fileName):
        os.mkdir(fileName)
    os.chdir(fileName)
    
    anpassInput = open("xtb-Anpass","w")
    with open(anpass_header,"r") as readHeader: #read header
        header = readHeader.readlines()
    for line in header:
        anpassInput.write(line) #write header
    with open(dispDat) as displacement:
        disp = displacement.readlines()
    for line in disp:
        anpassInput.write(f"{line.rstrip()}{next(iterData):20.12f}\n")
    with open(anpass_footer,"r") as foot: #read footer
        footer = foot.readlines()
    for line in footer:
        anpassInput.write(line) 
    anpassInput.close()
    os.system("/home/freu9584/bin/anpass-fit.e <xtb-Anpass> Anpass1.out")
    
    secondInput = open("AnpassSecond","w") 
    with open('xtb-Anpass',"r") as copyFile:
        anpassLines = copyFile.readlines()
    for line in anpassLines[:-4]:
        secondInput.write(line)
    secondInput.write(f"STATIONARY POINT\n")
    with open("fort.983","r") as statPoint:
        statData = statPoint.readline()
    secondInput.write(statData)
    secondInput.write(f"END OF DATA\n!FIT\n!END\n")
    secondInput.close()
    os.system(f"/home/freu9584/bin/anpass-fit.e <AnpassSecond> Anpass2.out")
    
    outputFile = open('IntderFile',"w")
    with open(intder_header1,"r") as headerFile:
        header = headerFile.readlines()
    data = list(map(lambda u:float(u),statData.split()[:-1]))
    disps = []
    for l in data:
        if l != 0.0:
            disps.append(l)
    for line in header[:-3]:
        outputFile.write(line)
        if "DISP" in line: break
    for t in range(len(disps)) :
        outputFile.write(f"{t+1: 5}{disps[t]:22.12f}\n")
    outputFile.write(f"    0\n")
    outputFile.close()
    os.system("/home/freu9584/bin/Intder2005.e <IntderFile> Intder.out")
    
    end_file = open("IntderFile2","w")
    with open(intder_header2,"r") as headerFile:
        header = headerFile.readlines()
        
    deriv = int(header[1].split()[3]) 

    w = 0
    for line in header:
        w+=1
        end_file.write(line)
        if line.strip() == "0": break
            
    with open('Intder.out',"r") as geomFile:
        geometry = geomFile.readlines()
    for line in geometry[-n_atoms:]:
        numbers = line.split()
        end_file.write(f"{float(numbers[0]):18.10f}{float(numbers[1]):19.10f}{float(numbers[2]):19.10f}\n")
    end_file.write(header[w+n_atoms])

    os.system("../../../sort_fort.sh")

    with open("sorted_fort.9903","r") as symmetryFile:
        symmetry = symmetryFile.readlines()
    columnCounter = 2 
    begin = False
    for line in symmetry[:]: 
        if begin == True:
            tempLine = line.split()
            if not(columnCounter == deriv) and not(tempLine[columnCounter]=="0"):
                columnCounter+=1
                end_file.write(f"    0\n")
            end_file.write(line)
        elif len(line.split()) ==1:
            begin = True
    end_file.write(f"    0\n")
    end_file.close()
    os.system("/home/freu9584/bin/Intder2005.e <IntderFile2> Intder2.out")
    
    spectroFile = open("SpectroFile","w")
    atomicNumIter = iter(at_num)
    with open(spectro_template,"r") as templateFile:
        template = templateFile.readlines()
    for line in template[:5]:
        spectroFile.write(line)
    with open("Intder2.out","r") as geomFile:
        geom = geomFile.readlines()
    for line in geom[16:16+n_atoms]:
        numbers = line.split()
        spectroFile.write(f"{next(atomicNumIter):5.2f}{float(numbers[0]):19.10f}{float(numbers[1]):19.10f}{float(numbers[2]):19.10f}\n")
    for line in template[5+n_atoms:]:
        spectroFile.write(line)
    spectroFile.close()
    
    num = iter([[15,15],[20,30],[24,40]])
    for p in range(3):
        files = next(num)
        with open(f"file{files[0]}","r") as originalFile:
            original = originalFile.readlines()
        copy = open(f"fort.{files[1]}","w")
        for line in original:
            copy.write(line)
        copy.close()
    os.system("/home/freu9584/bin/spectro.e <SpectroFile> Spectro.out")
    os.chdir("..")

def xtb_method(n_atoms, charge, structures, structure_file, at_nums):
    
    input_path = "xtb_files"
    os.system(f"mkdir {input_path}") #making inputs folder
    moleculeName = os.getcwd().split("/")[-2]
    all_files =[]
    
    with open(os.path.join(structure_file),"r") as geomfile:
        lines = geomfile.readlines()
    padding_zeros = len(str(structures)) # calculate the number of padding zeroes needed from the number of molecules
    first =1
    last =n_atoms+1
    
    for mol in range(structures):
        current_file = open(os.path.join(input_path,f"{moleculeName}-coord{mol:0{padding_zeros}}.dat"),"w") 
        all_files.append(os.path.join(input_path,f"{moleculeName}-coord{mol:0{padding_zeros}}.dat"))
        
        atom_count = -1
        current_file.write("$coord\n")
        for line in lines[first:last]:
            if line[0] !="#":
                atom_count += 1
                current_file.write(line.strip()+" "+ at_sym_dict[at_nums[atom_count]]+"\n")
        current_file.write("$end\n")
        first = last+1
        last = first+n_atoms
        current_file.close()
        
    os.system(f"echo {charge} > .CHRG")# specifying the charge of the molecule
    return all_files

def xtb_run(w):
    energy = 0
    output = w.split(".")[0]+'_output'
    outputName = os.path.join(os.getcwd(),output)
    os.system(f"/share/apps/xtb-6.3.2/bin/xtb -P 8 {w} > {outputName}")
    with open(outputName,"r") as output:
        out_lines = output.readlines()
        for line in out_lines:
            if ("TOTAL ENERGY" in line):
                energy = (float(line.split()[3]))
    return energy

def clear_files():
    os.system('rm mol* fort* opt*')  
    
def get_Spectro(spectro_file = "freq_files/Spectro.out"):
    r_equil = []
    r_alpha = []
    r_G =[]
    fundamental = []
    first = True
    BXS = 0
    BYS = 0
    BZS = 0
    result = np.array([])
    with open(spectro_file,"r") as spectro:
        lines = spectro.readlines()
    bond_bend = int(lines[14].split()[-1])+int(lines[15].split()[-1])
    for f,line in enumerate(lines):
        if "VIBRATIONALLY AVERAGED COORDINATES" in line:
            for r in range(bond_bend):
                r_equil.append(float(lines[f+13+r].split()[2]) if not "*" in lines[f+13+r].split()[2] else float(0))
                r_G.append(float(lines[f+13+r].split()[3]) if not "*" in lines[f+13+r].split()[3] else float(0))
                r_alpha.append(float(lines[f+13+r].split()[4]) if not "*" in lines[f+13+r].split()[4] else float(0))
        if "FUNDAMENTAL" in line:
            t=0
            while len(fundamental) < bond_bend:
                if not len(lines[f+t].split())==0 :
                    if "*" in lines[f+t].split()[-2]:
                            fundamental.append(float(0))
                    elif lines[f+t].split()[-2][-1].isdigit():
                            fundamental.append(float(lines[f+t].split()[-2]))
                t+=1
        if "BXS" in line and first :
            first = False
            BXS = float(lines[f+1].split()[0]) if not "*" in lines[f+1].split()[0] else float(0)
            BYS = float(lines[f+1].split()[1]) if not "*" in lines[f+1].split()[1] else float(0)
            BZS = float(lines[f+1].split()[2]) if not "*" in lines[f+1].split()[2] else float(0)
    if len(r_equil) == 0:
        for y in range(bond_bend):
            r_equil.append(float(0))
    if len(r_alpha) == 0:
        for y in range(bond_bend):
            r_alpha.append(float(0))
    if len(r_G) == 0:
        for y in range(bond_bend):
            r_G.append(float(0))
    if len(fundamental) == 0:
        for y in range(bond_bend):
            fundamental.append(float(0))
    result = np.hstack([r_equil,r_alpha,r_G,fundamental,BXS,BYS,BZS])
    return result

def xtb_write_parms(new_parms):
    parmIter = iter(new_parms)
    count =1
    global_parms = ["ks","kp","kd","ksd","kpd","kdiff","enscale","ipeashift","gam3s","gam3p","gam3d1",
                            "gam3d2","aesshift","aesexp","aesrmax","alphaj","a1","a2","s8","s9","aesdmp3",
                            "aesdmp5","kexp","kexplight"]
    element_parms = ["lev=","exp=","GAM=","GAM3=","KCNS=","KCNP=","DPOL=","QPOL=","REPA=","REPB=","POLYS=",
                               "POLYP=","LPARP=","LPARD="]
    with open("3param_gfn2-xtb.txt","r") as parm:
        parmlines = parm.readlines()
    new_file = open("param_gfn2-xtb.txt","w")
    for line in parmlines[:5]:
        new_file.write(line)
    for param in global_parms:
        new_file.write(f"{param.ljust(12)}   {float(next(parmIter)):10.5f}\n")
    new_file.write("$end\n$pairpar\n$end\n")
    for line in parmlines[len(global_parms)+8:]:
        if line.split()[0] in element_parms:
            new_file.write(f" {line.split()[0].ljust(6)}")
            if line.split()[0] == "lev=" or line.split()[0] == "exp=":
                if count <= 2:
                    new_file.write(f" {float(next(parmIter)):10.6f}\n")
                    count = count +1
                else:
                    new_file.write(f" {float(next(parmIter)):10.6f}  {float(next(parmIter)):10.6f}\n")
            else: new_file.write(f" {float(next(parmIter)):10.6f}\n")
        else: new_file.write(line)
    new_file.close()
    
def xtb_read_parms():
    count =0
    with open("3param_gfn2-xtb.txt","r") as parm_file:
        lines = parm_file.readlines()
    parm_list =[]
    for line in lines[5:]:
        if not "$" in line and not line.strip()[0:3] == "ao=":
            parm_list.append(float(line.split()[1]))
            if "exp=" in line or "lev=" in line:
                count +=1
                if count >2:
                    parm_list.append(float(line.split()[2]))
    return parm_list

def clear_dask():
    os.chdir("dask-worker-space")
    os.system("rm -r worker-*")
    os.chdir("..")

if __name__ == "__main__" :
    rmsd =[]
    global spec_count
    spec_count =0
    method_num, n_molec, n_atoms, charge, structures, energy_files, structure_files, at_num, coords, n_geoms, geoms, n_weights, weights, num_workers = read_input('main.inp')
    cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1,memory_limit="1GB",processes=False)
    client = Client(cluster)
    sturcture_files = np.array(structure_files)
    abinitio_energies = read_abinito(energy_files)

    if method_num == -14:
        parm_vals = xtb_read_parms()
    else:
        parm_labels, parm_vals = read_parms(sys.argv[1])
        for mol in range(n_molec):
            write_input(structure_files[mol],
                        structures[mol],
                        n_atoms[mol],at_num[mol],
                        method_num, mol, charge[mol])
    x, flag = leastsq(big_loop,parm_vals, method_num, epsfcn=1e-4)
    big_loop(x,method_num)
    fvec, energies = calc_fvec(structures, weights, n_geoms, geoms, method_num,n_atoms,at_num, True)
    if np.sum(n_geoms) > 0:
        print  ('FINAL RMSD ' + str(np.sqrt(np.mean(np.square(fvec[0:-np.sum(n_geoms)])))))
    else:
        print  ('FINAL RMSD  ' + str(np.sqrt(np.mean(np.square(fvec)))))
    print(spec_count)
    plt.plot(energies)
    plt.plot(abinitio_energies)
    plt.savefig('test.png', dpi=300)
    client.close()
    clear_dask()