
from scipy.optimize import minimize_scalar
import numpy as np
import warnings

def objective(omega, solver, tlin, dt):
    # Set the current omega value in the solver.
    solver.omega_1 = omega
    
    # Run the simulation.
    solution = solver.solve_ivp(dt=dt, tlin=tlin, x=0, apply_sinc=True, force_max_step_size=False)
    
    # Minimize w.r.t. last value
    last_value = np.abs(solution.y[2][-1])
    
    return last_value

def find_best_omega(solver, initial_bounds, tlin, dt, tolerance=1e-3):
    # Use minimize_scalar with the bounded method over the initial bounds.
    res = minimize_scalar(objective, args=(solver, tlin, dt), bounds=initial_bounds, method='bounded')
    
    best_omega = res.x
    best_res = res.fun

    # Warn if tolerance not achieved.
    if best_res > tolerance:
        warnings.warn("Did not reach tolerance.", UserWarning)
    
    return best_omega, best_res