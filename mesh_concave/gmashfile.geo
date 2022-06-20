// Gmsh project created on Fri Nov 13 14:58:52 2020
SetFactory("OpenCASCADE");
//+
Sphere(1) = {0.0, -0.25, 0, 0.5, -Pi/2, Pi/2, Pi};
Sphere(2) = {0.0, -0.25, 0, 0.48, -Pi/2, Pi/2, Pi};
BooleanDifference(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };
