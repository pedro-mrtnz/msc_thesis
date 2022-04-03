import os 
import subprocess
import numpy as np

import meshio
import gmsh

def create_box_mesh(xmin, xmax, ztop, zbot, lc):
    """_summary_

    Args:
        xmin (float): leftmost boundary
        xmax (float): rightmosht boundary
        ztop (float): uppermost boundary
        zbot (float): bottommost boundary
        lc   (float): mesh element size
    Return_
        mesh (meshio.Mesh): box-like mesh.
    """
    gmsh.initialize()
    model = gmsh.model 
    
    bl = model.geo.addPoint(xmin, zbot, 0, lc)
    br = model.geo.addPoint(xmax, zbot, 0, lc)
    tr = model.geo.addPoint(xmax, ztop, 0, lc)
    tl = model.geo.addPoint(xmin, ztop, 0, lc)
    
    lines = [
        model.geo.addLine(bl, br),
        model.geo.addLine(br, tr),
        model.geo.addLine(tr, tl),
        model.geo.addLine(tl, bl)
    ]
    curve_l = model.geo.addCurveLoop(lines)
    surface = model.geo.addPlaneSurface([curve_l])
    
    nx = np.ceil((xmax - xmin)/lc).astype(int)
    nz = np.ceil((ztop - zbot)/lc).astype(int)
    
    model.geo.mesh.setTransfiniteCurve(lines[0], nx)
    model.geo.mesh.setTransfiniteCurve(lines[1], nz)
    model.geo.mesh.setTransfiniteCurve(-lines[2], nx)
    model.geo.mesh.setTransfiniteCurve(-lines[3], nz)
    model.geo.mesh.setTransfiniteSurface(surface, 'Left')
    
    model.geo.synchronize()
    
    # Set some meshing options
    gmsh.option.setNumber('Mesh.RecombineAll', 1)
    gmsh.option.setNumber('Mesh.Algorithm', 5)
    gmsh.option.setNumber('Mesh.ElementOrder', 1)
    
    collect_physical_groups = {
        'Bottom': model.addPhysicalGroup(1, [lines[0]]),
        'Right' : model.addPhysicalGroup(1, [lines[1]]),
        'Top'   : model.addPhysicalGroup(1, [lines[2]]),
        'Left'  : model.addPhysicalGroup(1, [lines[3]]),
        'M1'    : model.addPhysicalGroup(2, [surface])
    }
    for label, group in collect_physical_groups.items():
        dim = 2 if label.startswith('M') else 1
        model.setPhysicalName(dim, group, label)
    
    gmsh.model.mesh.generate(dim=2)
    gmsh.write("mesh.msh")
    gmsh.finalize()

    mesh = meshio.read("mesh.msh")
    os.remove("mesh.msh")

    return mesh


def process_gmsh(mesh, filename='model'):
    """ 
    Here we run the mesh in an amenable way for SPECFEM to process it.
    
    Args:
        mesh     (meshio.Mesh): meshio we want to process.
        filename (str)        : name of file that's going to be saved. 
    """
    current_dir = os.getcwd()
    if not os.path.exists('MESH/'):
        os.system('mkdir -p MESH/')
    #os.system('rm -rf MESH/*')
    os.chdir('MESH')

    msh_filename = f'{filename}.msh'
    meshio.write(msh_filename, mesh, file_format='gmsh22', binary=False)

    # .sh job
    sh_filename = 'process_gmsh.sh'
    with open(sh_filename, 'w') as f:
        f.write("")
        f.write(
    f"""echo "Exporting to SPECFEM mesh files"
echo

# Convert to SPECFEM format files
python /Users/Pedro/specfem2d/utils/Gmsh/LibGmsh2Specfem_convert_Gmsh_to_Specfem2D_official.py {msh_filename} -t F -l A -b A -r A

# Checks exit code
if [[ $? -ne 0 ]]; then exit 1; fi

echo "Done!"
    """)

    subprocess.call(['sh', f'./{sh_filename}'])

    os.chdir('..')
    assert(os.getcwd() == current_dir)