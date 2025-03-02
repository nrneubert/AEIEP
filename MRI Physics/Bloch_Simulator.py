
import warnings
import numpy as np
from scipy.integrate import solve_ivp

class Bloch_Simulator() : 
                                        
    def __init__(self, gamma = 2.675 * 1e8,         # rad/(T*s) 
                    T1 = 1000*1e-3, T2=1000*1e-3,   # s 
                    M0 = 1.0, 
                    phi = 0.0, 
                    omega_1 = 0.25 * 1e4,           # 1/s 
                    delta_omega = 0.0,              # 1/s 
                    nz = 1.0,                       # nmb. 
                    tw = 1 * 1e-3,                  # s 
                    G = None, 
                    lower_bound=-np.inf, upper_bound=np.inf
                )  -> None:
        # Constants
        self.delta_x = 2 * 1e-3                     # m
        # Parameters
        self.gamma = gamma
        self.T1 = T1
        self.T2 = T2
        self.M0 = M0
        self.phi = phi
        self.omega_1 = omega_1
        self.delta_omega = delta_omega
        self.nz = nz
        self.tw = tw
        if(G) : self.G = 2*np.pi/(self.gamma*self.tw*self.delta_x)
        else :  self.G = G
        # Pulse bounds
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        # Logged information
        self.last_solution = None

    def _get_Beff(self, t: float, x: float, apply_sinc: bool)  -> np.ndarray:
        B1 = self._get_omega1(t) / self.gamma
        Beff = np.zeros(3)
        
        Beff[0] += B1*np.cos(self.phi)
        Beff[1] += B1*np.sin(self.phi)
        Beff[2] += self.delta_omega/self.gamma

        if(apply_sinc) : Beff *= np.sin(2*np.pi*t/self.tw)/(2*np.pi*t/self.tw)

        if(x) : Beff[2] += self.G * x

        return Beff


    def _get_omega1(self, t: float) -> float: 
        
        if(t >= self.lower_bound and t <= self.upper_bound) : 
            return self.omega_1
        else : 
            return 0
    
    def _diff_eqn(self, t: float, M: list, x: float, apply_sinc: bool) -> np.ndarray: 
        
        
        M = np.array(M)
        Beff = self._get_Beff(t, x, apply_sinc)
        
        # Precession
        dMdt = - self.gamma * np.cross(M, Beff)

        # Relaxation terms
        dMdt[0] += - 1/self.T2 * M[0]
        dMdt[1] += - 1/self.T2 * M[1]
        dMdt[2] += - 1/self.T1 * (M[2] - self.M0)

        return dMdt

    def solve_ivp(self, dt: float, tlin: list, x: float, apply_sinc=True, force_max_step_size=False) : # -> OdeResult
        M0_init = np.array([0.0, 0.0, self.M0])
        step_size = dt if force_max_step_size else None

        print(step_size)

        solution = solve_ivp(fun=self._diff_eqn, 
                             y0=M0_init, 
                             t_span=(min(tlin), max(tlin)), 
                             t_eval=tlin, 
                             max_step=step_size,
                             args=(x, apply_sinc,)
                             )
                    
        self.last_solution = solution

        if not solution.success : warnings.warn(solution.message, UserWarning)

        return solution
        