"""
Gmsh script which can serve as inspiration for the multilayer problem.
"""
import gmsh

def generate_mesh():
    ysup = 0.6
    yinf = 0.3
    
    gmsh.initialize()
    gmsh.model.add("horiz_ref")
    
    # Siple rectangular geomtry
    lc = .15
    gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc, 1)
    gmsh.model.geo.addPoint(1.0, 0.0, 0.0, lc, 2)
    gmsh.model.geo.addPoint(1.0, 1.0, 0.0, lc, 3)
    gmsh.model.geo.addPoint(0.0, 1.0, 0.0, lc, 4)
    
    # gmsh.model.geo.addPoint(0.0, yinf, 0.0, lc, 5)
    # gmsh.model.geo.addPoint(1.0, yinf, 0.0, lc, 6)
    # gmsh.model.geo.addPoint(1.0, ysup, 0.0, lc, 7)
    # gmsh.model.geo.addPoint(0.0, ysup, 0.0, lc, 8)
    
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)
    # gmsh.model.geo.addLine(5, 5, 6)  # Inferior horizontal line
    # gmsh.model.geo.addLine(6, 7, 8)  # Superior horizontal line
    
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 5)
    gmsh.model.geo.addPlaneSurface([5], 6)

    gmsh.model.geo.synchronize()
    
    gmsh.model.mesh.field.add("Box", 1)
    gmsh.model.mesh.field.setNumber(1, "VIn", lc / 30)
    gmsh.model.mesh.field.setNumber(1, "VOut", lc)
    gmsh.model.mesh.field.setNumber(1, "XMin", 0.0)
    gmsh.model.mesh.field.setNumber(1, "XMax", 1.0)
    gmsh.model.mesh.field.setNumber(1, "YMin", yinf)
    gmsh.model.mesh.field.setNumber(1, "YMax", ysup)
    gmsh.model.mesh.field.setNumber(1, "Thickness", 0.3)
    
    gmsh.model.mesh.setRecombine(2, 6)
    
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh.option.setNumber('Mesh.ElementOrder', 1)
    
    gmsh.model.mesh.generate(2)
    gmsh.write("mesh.msh")
    gmsh.finalize()
    
    
    
