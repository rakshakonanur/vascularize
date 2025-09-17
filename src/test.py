import pyvista as pv
from svv.domain.domain import Domain
from svv.tree.tree import Tree
import numpy as np

# Creating the Tissue Domain
cube = Domain(pv.Cube())
cube.create()
cube.solve()
cube.build()

# Creating the Vascular Tree Object
t = Tree()
t.set_domain(cube)
t.set_root(start=np.array([[1.0, 1.0, 0.5]]))  # Setting the root at the center of the cube
t.n_add(50)

# Visualizing the Tree and Domain
t.show(plot_domain=True)