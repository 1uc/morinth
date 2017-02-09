import numpy as np

class SourceTerm:
    def __init__(self):
        self.needs_edge_source = hasattr(self, "edge_source")
        self.needs_volume_source = hasattr(self, "volume_source")

class CenteredSourceTerm(SourceTerm):
    def __init__(self, model):
        super().__init__()

        self.model = model

    def volume_source(self, u_bar, u_left=None, u_right=None):
        """Compute the balanced source term of the Euler equations.

        Arguments:
              u_bar : cell-average of the conserved variables
             u_left : ---
            u_right : ---
        """
        return self.model.source(u_bar)


class BalancedSourceTerm(SourceTerm):
    def __init__(self, grid, model, equilibrium, order=4):
        super().__init__()

        self.grid = grid
        self.model = model
        self.equilibrium = equilibrium
        self.order = order

    def edge_source(self, u_bar, u_left, u_right, axis):
        """Compute the balanced source term of the Euler equations.

        Parameters
        ----------
               u_bar : array_like
                       cell-average of the conserved variables

              u_left : array_like
                       point-value of the conserved variables at the left
                       boundary

             u_right : array_like
                       point-value of the conserved variables at the right
                       boundary

                axis : int
                       direction for which to compute the source term
        """
        assert axis == 0

        if self.order == 2:
            return self.edge_source_o2(u_bar)
        elif self.order == 4:
            return self.edge_source_o4(u_bar, u_left, u_right)
        else:
            raise Exception("Invalid order [{:d}].".format(self.order))


    def edge_source_o4(self, u_bar, u_left, u_right):

        x_ref = self.grid.cell_centers[...,0]
        x_right = self.grid.edges[1:,...,0]
        x_left = self.grid.edges[:-1,...,0]

        rho_point, p_point, T_point, _, _ = self.equilibrium.point_values(u_bar)

        rho_left, p_left = self.equilibrium.extrapolate(p_point, T_point, x_ref, x_left)
        rho_right, p_right = self.equilibrium.extrapolate(p_point, T_point, x_ref, x_right)

        dudt = np.zeros_like(u_bar)
        dudt[1,...] = (p_right - p_left)/self.grid.dx
        dudt[3,...] = 0.0

        return dudt
