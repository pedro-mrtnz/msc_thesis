import gmsh
import numpy as np

def generate_mesh():
    lc = 10.0
    xmin, xmax = 0, 2000
    zmin, zmax = -3000, 0
    
    zmin_mult = -2000.0
    zmax_mult = -1500.0
    
    L_mult = zmax_mult - zmin_mult
    L1 = zmax - zmax_mult
    L2 = zmin_mult - zmin
    
    # Resolution factor for the multilayer
    mres = 4

    gmsh.initialize()
    gmsh.model.add("struct_transf")
    model = gmsh.model

    lb = model.geo.addPoint(xmin, zmin, 0, lc)  # Left bottom
    rb = model.geo.addPoint(xmax, zmin, 0, lc)  # Right bottom
    rt = model.geo.addPoint(xmax, zmax, 0, lc)  # Right top
    lt = model.geo.addPoint(xmin, zmax, 0, lc)  # Left top

    p1 = model.geo.addPoint(xmin, zmin_mult, 0, lc)  # Multilayer zone
    p2 = model.geo.addPoint(xmax, zmin_mult, 0, lc)
    p3 = model.geo.addPoint(xmax, zmax_mult, 0, lc)
    p4 = model.geo.addPoint(xmin, zmax_mult, 0, lc)

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

    n_top  = np.ceil(L1/lc).astype(int) 
    n_mult = np.ceil(mres * L_mult/lc).astype(int)
    n_bot  = np.ceil(L2/lc).astype(int)
    
    # nz = n_top + n_mult + n_bot
    # nx = np.ceil((xmax - xmin)/lc).astype(int)

    model.geo.mesh.setTransfiniteCurve(p2l['lt_p4'], n_top, 'Progression', -1.01)
    model.geo.mesh.setTransfiniteCurve(p2l['p3_rt'], n_top, 'Progression', 1.01)
    model.geo.mesh.setTransfiniteCurve(p2l['p4_p1'], n_mult, 'Progression', 1.0)
    model.geo.mesh.setTransfiniteCurve(p2l['p2_p3'], n_mult, 'Progression', 1.0)
    model.geo.mesh.setTransfiniteCurve(p2l['p1_lb'], n_bot, 'Progression', 1.01)
    model.geo.mesh.setTransfiniteCurve(p2l['rb_p2'], n_bot, 'Progression', -1.01)

    model.geo.mesh.setTransfiniteSurface(surf_mult, 'Left')
    model.geo.mesh.setTransfiniteSurface(surf_bot, 'Left')
    model.geo.mesh.setTransfiniteSurface(surf_top, 'Left')

    gmsh.model.geo.synchronize()

    # collect_physical_groups = {
    #     'Bottom': model.addPhysicalGroup(1, [p2l['lb_rb']]),
    #     'Right' : model.addPhysicalGroup(1, [p2l['rb_p2'], p2l['p2_p3'], p2l['p3_rt']]),
    #     'Top'   : model.addPhysicalGroup(1, [p2l['rt_lt']]),
    #     'Left'  : model.addPhysicalGroup(1, [p2l['lt_p4'], p2l['p4_p1'], p2l['p1_lb']]),
    #     'M1'    : model.addPhysicalGroup(2, [surf_bot, surf_mult, surf_top])
    # }
    # for label, group in collect_physical_groups.items():
    #     dim = 2 if 'M' == label[0] else 1
    #     model.setPhysicalName(dim, group, label)

    gmsh.model.mesh.generate(dim=2)
    gmsh.write("mesh.msh")
    gmsh.finalize()