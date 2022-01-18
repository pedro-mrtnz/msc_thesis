import os
import subprocess
import numpy as np
import pandas as pd

import pygmsh
import meshio
import gmsh

###################################################################################
## HELPER FUNCTIONS 
###################################################################################

def print_script(geom, delete=True):
    filename = 'current_script.geo_unrolled'
    geom.save_geometry(filename)
    with open(filename, 'r') as f:
        lines = f.readlines()
        script = ''
        for l in lines:
            script += l
        
        print('CURRENT GEOM SCRIPT:')
        print(script)
    
    os.remove(filename)
    
def print_mesh_info(mesh):
    print('MESH RESULTS:')
    print(f"{'  Points':<20} {len(mesh.points)}")
    print(f"{'  Cells':<20} {len(mesh.cells_dict)}")
    for name, array in mesh.cells_dict.items():
        print(f"{'    '+ name:<20} {len(array)}")
    print(f"{'  Point data':<20} {len(mesh.point_data)}")
    print(f"{'  Cell data':<20} {len(mesh.cell_data_dict)}")
    for name, data in mesh.cell_data_dict.items():
        print('    ' + name)
        for s, array in data.items():
            print(f"{'      '+ s:<20} {len(array)}")
    print(f"{'  Field data':<20} {len(mesh.field_data)}")
    for name, array in mesh.field_data.items():
        print(f"{'    '+ name:<20} {len(array)}")
    print(f"{'  Cell sets':<20} {len(mesh.cell_sets)}")
    for name, array in mesh.cell_sets.items():
        print(f"{'    '+ name:<20} {len(array)}")
        
    return mesh.points, mesh.cells_dict, mesh.point_data, mesh.cell_data_dict, mesh.field_data, mesh.cell_sets


###################################################################################
## DEFINE MESH 
###################################################################################

def create_multilayer_mesh(xmin, xmax, ytop, ybot, lc, N, verbose=True):
    """Creates a multilayer mesh with sharp boundaries, where all the layers have the same size.

    Args:
        xmin (float): minimum -left- point along x axis
        xmax (float): maximum -right-point along x axis
        ytop (float): top point along y axis 
        ybot (float): bottom point (OF THE MESH) along y axis
        N    (int)  : number of layers in the multilayer
        lc   (float): element size
        
    Return: mesh info
    """
    # Fixme: use TrnasfiniteCurve and Surface to make it completely structured.
    # Length of the layers
    L = np.abs(ytop - ybot)/N
    
    geom = pygmsh.geo.Geometry()
    geom.__enter__()

    # Set some meshing options
    gmsh.option.setNumber('Mesh.RecombineAll', 1)  # Creates quadrangles instead of triangles
    gmsh.option.setNumber('Mesh.Algorithm', 8)     # 1: MeshAdapt, 2: Automatic, 3: Initial mesh only, 5: Delaunay, 6: Frontal-Delaunay, 7: BAMG, 8: Frontal-Delaunay for Quads, 9: Packing of Parallelograms)
    gmsh.option.setNumber('Mesh.ElementOrder', 1)  # 1: linear order, 2: quadratic...
    #msh.option.setNumber('Geometry.ExtrudeReturnLateralEntities', 1)
    #gmsh.option.setNumber('Geometry.AutoCoherence', 1)
    
    for i in range(N):
        ybot = ytop - L
        if i == 0:  # Layer 1
            # Corners ANTICLOCKWISE
            corners_new = [geom.add_point([xmin, ybot, 0], lc),  # Bottom left
                           geom.add_point([xmax, ybot, 0], lc),  # Bottom right
                           geom.add_point([xmax, ytop, 0], lc),  # Top right
                           geom.add_point([xmin, ytop, 0], lc)]  # Top left
            
            lines_ = []
            for pt1, pt2 in zip(corners_new, corners_new[1:] + [corners_new[0]]):
                lines_.append(geom.add_line(pt1, pt2))
            # FOR WHEN WE DEFINE THE PHYSCAL GROUPS
            LEFT_LINES  = [lines_[3]]  # we´ll be appending to this 
            RIGHT_LINES = [lines_[1]]  # we´ll be appending to this 
            TOP_LINE    = lines_[2]
            
            curve_loop_ = geom.add_curve_loop(lines_)
            SURFACES = [geom.add_plane_surface(curve_loop_)]
            
        else:
            pt1_new = geom.add_point([xmin, ybot, 0], lc)
            pt2_new = geom.add_point([xmax, ybot, 0], lc)

            corners_old = corners_new.copy()
            corners_new = [pt1_new, pt2_new, corners_old[1], corners_old[0]]

            lines2points = {l: l.points for l in lines_[:]}  # Map bw lines and its points from previous layer
            lines_ = []                                      # New layer lines
            for pt1, pt2 in zip(corners_new, corners_new[1:] + [corners_new[0]]):
                if [pt2, pt1] in lines2points.values():
                    line = [k for k, v in lines2points.items() if v == [pt2, pt1]][0]
                    lines_.append(-line)
                else:
                    lines_.append(geom.add_line(pt1, pt2))
            LEFT_LINES.append(lines_[3])
            RIGHT_LINES.append(lines_[1])

            curve_loop_ = geom.add_curve_loop(lines_)
            SURFACES.append(geom.add_plane_surface(curve_loop_))
            
        ytop = ybot
        
    BOTTOM_LINE = lines_[0]
    RIGHT_LINES.sort(key = lambda x: x._id, reverse=True)
    
    # Add physical groups
    geom.add_physical(RIGHT_LINES, label='Right')
    geom.add_physical(TOP_LINE, label='Top')
    geom.add_physical(LEFT_LINES, label='Left')
    geom.add_physical(BOTTOM_LINE, label='Bottom')
    for i, s in enumerate(SURFACES):
        geom.add_physical(s, label=f'M{i+1}')
    
    # Generate mesh
    mesh = geom.generate_mesh(verbose=True)
    geom.save_geometry('nan.msh')
    mesh = meshio.read('nan.msh')
    os.remove('nan.msh')
    
    if verbose:
        print_script(geom)
        print_mesh_info(mesh)
    
    return mesh


def create_box_mesh(xmin, xmax, ztop, zbot, lc):
    """
    Creates a square mesh with only one physical group M1 (a box) that
    serves as a background onto which define the different domains using
    the tomography file. I.e. no sharp boundaries.
    """
    geom = pygmsh.geo.Geometry()
    model = geom.__enter__()
    
    # Set some meshing options
    gmsh.option.setNumber('Mesh.RecombineAll', 1) 
    gmsh.option.setNumber('Mesh.Algorithm', 8)    
    gmsh.option.setNumber('Mesh.ElementOrder', 1)

    # corners = [
    #     model.add_point([xmin, zbot, 0], lc),
    #     model.add_point([xmax, zbot, 0], lc),
    #     model.add_point([xmax, ztop, 0], lc),
    #     model.add_point([xmin, ztop, 0], lc)   
    # ]
    
    # lines = []  # [bottom, right, top, left]
    # for pt1, pt2 in zip(corners, corners[1:] + [corners[0]]):
    #     lines.append(model.add_line(pt1, pt2))
    # curve_l = model.add_curve_loop(lines)
    # surface = model.add_plane_surface(curve_l)
    
    # model.synchronize()
    
    nx = np.ceil((xmax - xmin)/lc).astype(int)
    nz = np.ceil((ztop - zbot)/lc).astype(int)
    
    rect = model.add_rectangle(xmin, xmax, zbot, ztop, 0, lc)
    surface = rect.surface
    
    boundaries = ['Bottom', 'Right', 'Top', 'Left']
    for line, boundary in zip(rect.lines, boundaries):
        model.add_physical(line, label=boundary)
    model.add_physical(surface, label='M1')
    
    geom.generate_mesh(dim=2, verbose=True)
    gmsh.write('nan.msh')
    gmsh.clear()
    geom.__exit__()
    
    mesh = meshio.read('nan.msh')
    os.remove('nan.msh')
    
    return mesh, nx, nz


def create_fine_box_mesh(xmin, xmax, ztop, zbot, lc, uneven_dict, mres):
    """
    Creates regular box model, but which contains a finer zone. E.g. this finer
    zone is where the multilayer will be defined.
    
    New args:
        uneven_dict (dict) : dict specifying domain ids along with its sizes. It also
                             contains the size of the multilayer L_mult. If None, we get
                             the otherwise regular grid.  
        mres        (float): resolution factor with which increase the resolution in 
                             multilayer w.r.t. the default L_mult/N resolution.
    
    Returns: mesh 
    
                (LT)--------(RT)
                |              | 
                |              |
                |              |
                |              |
                (P4)--------(P3)
                |  FINER MESH  | 
                (P1)--------(P2)
                |              | 
                |              |
                |              |
                |              |
                (LB)--------(RB)
        
    """
    if uneven_dict is None:
        # The otherwise regular grid is obtained
        return create_box_mesh(xmin, xmax, ztop, zbot, lc)
    else:
        L1, L_mult, L2  = uneven_dict[1], uneven_dict['L_mult'], uneven_dict[85]
        assert L1 + L_mult + L2 == ztop - zbot, "The sum of the domain ids doesn't amount to the size of the mesh"
        ztop_mult = ztop - L1
        zbot_mult = zbot + L2

        gmsh.initialize()

        gmsh.option.setNumber('Mesh.RecombineAll', 1) 
        gmsh.option.setNumber('Mesh.Algorithm', 6)    
        gmsh.option.setNumber('Mesh.ElementOrder', 1)
        # gmsh.option.setNumber('Mesh.Smoothing', 100)

        model = gmsh.model

        lb = model.geo.addPoint(xmin, zbot, 0, lc)  # Left bottom
        rb = model.geo.addPoint(xmax, zbot, 0, lc)  # Right bottom
        rt = model.geo.addPoint(xmax, ztop, 0, lc)  # Right top
        lt = model.geo.addPoint(xmin, ztop, 0, lc)  # Left top

        p1 = model.geo.addPoint(xmin, zbot_mult, 0, lc)  # Multilayer zone
        p2 = model.geo.addPoint(xmax, zbot_mult, 0, lc)
        p3 = model.geo.addPoint(xmax, ztop_mult, 0, lc)
        p4 = model.geo.addPoint(xmin, ztop_mult, 0, lc)

        # Points to lines mapping
        p2l = {
            'lb_rb': model.geo.addLine(lb, rb),
            'rb_p2': model.geo.addLine(rb, p2),
            'p2_p3': model.geo.addLine(p2, p3),
            'p3_rt': model.geo.addLine(p3, rt),
            'rt_lt': model.geo.addLine(rt, lt),
            'lt_p4': model.geo.addLine(lt, p4),
            'p4_p1': model.geo.addLine(p4, p1),
            'p1_lb': model.geo.addLine(p1, lb),
            'p1_p2': model.geo.addLine(p1, p2),
            'p3_p4': model.geo.addLine(p3, p4),
        }

        lines_mult = [p2l['p1_p2'], p2l['p2_p3'], p2l['p3_p4'], p2l['p4_p1']]   # Multilayer 
        lines_bot  = [p2l['lb_rb'], p2l['rb_p2'], -p2l['p1_p2'], p2l['p1_lb']]  # Bottom layer
        lines_top  = [-p2l['p3_p4'], p2l['p3_rt'], p2l['rt_lt'], p2l['lt_p4']]  # Top layer

        cl_mult = model.geo.addCurveLoop(lines_mult)
        cl_bot  = model.geo.addCurveLoop(lines_bot)
        cl_top  = model.geo.addCurveLoop(lines_top)

        surf_mult = model.geo.addPlaneSurface([cl_mult])
        surf_bot  = model.geo.addPlaneSurface([cl_bot])
        surf_top  = model.geo.addPlaneSurface([cl_top])

        n_top  = np.ceil(0.6 * L1/lc).astype(int) 
        n_mult = np.ceil(mres * L_mult/lc).astype(int)
        n_bot  = np.ceil(0.4 * L2/lc).astype(int)
        
        nz = n_top + n_mult + n_bot
        nx = np.ceil((xmax - xmin)/lc).astype(int)

        model.geo.mesh.setTransfiniteCurve(p2l['lt_p4'], n_top, 'Progression', -1.01)
        model.geo.mesh.setTransfiniteCurve(p2l['p3_rt'], n_top, 'Progression', 1.01)
        model.geo.mesh.setTransfiniteCurve(p2l['p4_p1'], n_mult, 'Progression', 1.0)
        model.geo.mesh.setTransfiniteCurve(p2l['p2_p3'], n_mult, 'Progression', 1.0)
        model.geo.mesh.setTransfiniteCurve(p2l['p1_lb'], n_bot, 'Progression', 1.01)
        model.geo.mesh.setTransfiniteCurve(p2l['rb_p2'], n_bot, 'Progression', -1.01)

        model.geo.mesh.setTransfiniteSurface(surf_mult, 'Left')
        model.geo.mesh.setTransfiniteSurface(surf_bot, 'Left')
        model.geo.mesh.setTransfiniteSurface(surf_top, 'Left')

        model.geo.synchronize()
        
        # model.mesh.field.add("Box", 1)
        # model.mesh.field.setNumber(1, "VIn", lc/2)
        # model.mesh.field.setNumber(1, "VOut", lc)
        # model.mesh.field.setNumber(1, "XMin", xmin)
        # model.mesh.field.setNumber(1, "XMax", xmax)
        # model.mesh.field.setNumber(1, "YMin", zbot_mult)
        # model.mesh.field.setNumber(1, "YMax", ztop_mult)
        # model.mesh.field.setNumber(1, "Thickness", 200)

        # model.mesh.field.setAsBackgroundMesh(1)

        # gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        # gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        # gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        gmsh.option.setNumber('Mesh.RecombineAll', 1)
        gmsh.option.setNumber('Mesh.Algorithm', 5)
        gmsh.option.setNumber('Mesh.ElementOrder', 1)

        collect_physical_groups = {
            'Bottom': model.addPhysicalGroup(1, [p2l['lb_rb']]),
            'Right' : model.addPhysicalGroup(1, [p2l['rb_p2'], p2l['p2_p3'], p2l['p3_rt']]),
            'Top'   : model.addPhysicalGroup(1, [p2l['rt_lt']]),
            'Left'  : model.addPhysicalGroup(1, [p2l['lt_p4'], p2l['p4_p1'], p2l['p1_lb']]),
            'M1'    : model.addPhysicalGroup(2, [surf_bot, surf_mult, surf_top])
        }
        for label, group in collect_physical_groups.items():
            dim = 2 if 'M' == label[0] else 1
            model.setPhysicalName(dim, group, label)

        gmsh.model.mesh.generate(dim=2)
        gmsh.write("mesh.msh")
        gmsh.finalize()
        
        mesh = meshio.read("mesh.msh")
        os.remove("mesh.msh")
        
        return mesh  #, nx, nz


###################################################################################
## MATERIAL FILE 
###################################################################################

def get_materials(path2refl, rho_ini, vp_ini, verbose=True):
    """
    Creates the material file from from the reflectivity profile. It assumes a linear 
    relationship between velocity and density: velocity = alpha * density
    
    Args:
        path2refl (str)  : path to the reflectivity profile
        rho_ini   (float): density of the first layer 
        vp_ini    (float): velocity of the first layer  
    
    Returns: 
        File is saved under the name 'material_file.txt', and returns the number of layers.
    """
    vs, Q_mu, Q_kappa = 0, 9999, 9999
    mat_id = 1
    alpha = 1.5
    if verbose:
        print('MATERIAL LAYER 1:')
        print(f"{'  Rho, Vp, Vs': <15} = {rho_ini}, {vp_ini}, {vs}")
        print(f"{'  Q_mu, Q_kappa': <15} = {Q_mu}, {Q_kappa}")
        print(f"{'  Material ID': <15} = {mat_id}")
        print(f"{'  Cte vp-rho': <15} = {alpha}")
        
    alpha = 1.5
    refls = np.loadtxt(path2refl)
    if not os.path.exists('MESH/'):
        os.system('mkdir -p MESH/')
    print('\nWritting down material_file.txt!\n')
    with open('MESH/material_file.txt', 'w') as f:
        domain_id = 1
        f.write(f"{domain_id} {mat_id} {rho_ini:.2f} {vp_ini:.2f} {vs} {0} {0} {Q_kappa} {Q_mu} 0 0 0 0 0 0\n")
        for R in refls:
            rho_new = np.sqrt(-(R + 1)/(R - 1) * vp_ini * rho_ini / alpha)
            vp_new  = alpha * rho_new - 200
            domain_id += 1
            f.write(f"{domain_id} {mat_id} {rho_new:.2f} {vp_new:.2f} {vs} {0} {0} {Q_kappa} {Q_mu} 0 0 0 0 0 0\n")
            
            rho_ini, vp_ini = rho_new, vp_new
    
    return len(refls) + 1


###################################################################################
## THE MODEL! 
###################################################################################

def process_gmsh(mesh, filename='model'):
    """
    Here we run the mesh in a amenable way for SPECFEM to process it.
    
    ! Fixme: BOUNDARY CONDITIONS?
    ! Fixme: INCLUDE HERE THE MATERIAL FILES?
    """
    current_dir = os.getcwd()
    if not os.path.exists('MESH/'):
        os.system('mkdir -p MESH/')
    #os.system('rm -rf MESH/*')
    os.chdir('MESH')

    # Save mesh as vtk file
    vtk_filename = f'{filename}.vtu'
    meshio.write(vtk_filename, mesh)

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