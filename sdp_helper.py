"""
This module contains helper functions for the SDP for two and four coherent
states
"""

import math
import numpy as np
import cvxpy as cp
from typing import List
#import mosek


def coh_state_vector(alpha: complex, n: int) -> np.array:

    """
    This function generates an array that represents a bra vector for a coherent
    state in the Fock basis. Each entry represents the amplitude for the
    corresponding Fock state. Since we can't have a vector with infinite
    entries, we cut off after Fock state |n>.

    Parameters
    ----------
        alpha : complex
            Eigenvalue of the coherent state (complex number).
        n : int
            The last fock that is represented.

    Returns
    -------
        state_vector : np.array
            The array represents the coherent state as a row vector.
    """

    vec = np.array([])

    for i in range(n+1):
        ampl = np.exp(-(1/2)*abs(alpha)**2) \
            *(alpha**i/math.sqrt(math.factorial(i)))
        vec = np.append(vec, ampl)

    return vec.reshape((len(vec), 1))


def density_matrix(state_vector: np.array) -> np.array:

    """
    This function receives a numpy array, which represent a bra vector and
    returns a numpy array, that represents the corresponding density matrix of
    the state.

    Parameters
    ----------
        state_vector : np.array
            represents the bra state vector

    Returns
    -------
        density_matrix : np.array
            represents the density matrix of the state
    """

    return state_vector.dot(state_vector.transpose().conjugate())


def sdp_primal_2_states(
        alpha_1: complex, 
        alpha_2: complex,
        xi_1: float,
        n : int,
        max_iters: int = 50000,
        zero_operators: List[str] = [],
    ) -> (float, float, float, np.array, np.array, np.array):

    """
    This function does the SDP for two coherent states with arbitrary input
    probabilites

    Parameters
    ----------
        alpha_1: np.array
        alpha_2: np.array
        xi_1: float
            Input probability of alpha_1 with 0 <= xi_1 <= 1

    Returns
    -------
        p_corr: float
        p_err: float
        Pi_1: np.array
        Pi_2: np.array
    """

    xi_2 = 1 - xi_1

    vec_state_1 = coh_state_vector(alpha_1, n)
    dens_matr_state_1 = density_matrix(vec_state_1)

    vec_state_2 = coh_state_vector(alpha_2, n)
    dens_matr_state_2 = density_matrix(vec_state_2)

    # Define the variables (POVM elements)
    Pi_1 = cp.Variable((n+1, n+1), hermitian=True)
    Pi_2 = cp.Variable((n+1, n+1), hermitian=True)

    constraints = [
        np.identity(n+1) - Pi_1 - Pi_2 >> 0,
        Pi_1 >> 0,  # Pi_1 is positive semidefinite
        Pi_2 >> 0,  # Pi_2 is positive semidefinite
        dens_matr_state_1 @ Pi_2 == 0,
        dens_matr_state_2 @ Pi_1 == 0,
    ]
    if "Pi_1" in zero_operators:
        constraints.append(Pi_1 << 0)
    if "Pi_2" in zero_operators:
        constraints.append(Pi_2 << 0)

    # objective (p_corr)
    objective = cp.real(cp.trace(dens_matr_state_1 @ Pi_1))*xi_1 \
        + cp.real(cp.trace(dens_matr_state_2 @ Pi_2))*xi_2

    # solve the problem
    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve(
        solver="SCS", 
        eps=1e-25, 
        max_iters=max_iters,
    )

    return [
        np.trace(np.matmul(Pi_1.value, dens_matr_state_1))*xi_1 \
            + np.trace(np.matmul(Pi_2.value, dens_matr_state_2))*xi_2,  # p_corr
        np.trace(np.matmul(Pi_2.value, dens_matr_state_1))*xi_1 \
            + np.trace(np.matmul(Pi_1.value, dens_matr_state_2))*xi_2,  # p_err
        Pi_1.value,  # Pi_1 obtained from SDP
        Pi_2.value,  # Pi_2 obtained from SDP
    ]


def sdp_primal_4_states(
        alpha_1: complex, 
        alpha_2: complex,
        alpha_3: complex, 
        alpha_4: complex,
        xi_1: float,
        xi_2: float,
        xi_3: float,
        xi_4: float,
        n : int,
        zero_operators: List[str] = [],
    ) -> (float, float, float, np.array, np.array, np.array):

    """
    This function does the SDP for four coherent states with arbitrary input
    probabilites xi_i.

    Parameters
    ----------
        alpha_1 : complex
            eigenvalue of state 1
        alpha_2 : complex
            eigenvalue of state 2
        alpha_3 : complex
            eigenvalue of state 3
        alpha_4 : complex
            eigenvalue of state 4
        xi_1 : float
            input probablity of state 1
        xi_2 : float
            input probablity of state 2
        xi_3 : float
            input probablity of state 3
        xi_4 : float
            input probablity of state 4
        n : int
            The last fock that is represented.
        zero_operators: List[str] = []
            Operators which are forced to be zero matrices. Enter strings as
            "Pi_1" for example.

    Returns
    -------
        p_corr: float
        Pi_1: np.array
        Pi_2: np.array
        Pi_3: np.array
        Pi_4: np.array
    """

    # all probabilities must add up to 1, otherwise error is raised
    if np.round(xi_1 + xi_2 + xi_3 + xi_4, 5) != 1:
        print(xi_1, xi_2, xi_3, xi_4)
        print(xi_1 + xi_2 + xi_3 + xi_4)
        raise ValueError

    vec_state_1 = coh_state_vector(alpha_1, n)
    dens_matr_state_1 = density_matrix(vec_state_1)

    vec_state_2 = coh_state_vector(alpha_2, n)
    dens_matr_state_2 = density_matrix(vec_state_2)

    vec_state_3 = coh_state_vector(alpha_3, n)
    dens_matr_state_3 = density_matrix(vec_state_3)

    vec_state_4 = coh_state_vector(alpha_4, n)
    dens_matr_state_4 = density_matrix(vec_state_4)

    # Define the variables (POVM elements)
    Pi_1 = cp.Variable((n+1, n+1), hermitian=True)
    Pi_2 = cp.Variable((n+1, n+1), hermitian=True)
    Pi_3 = cp.Variable((n+1, n+1), hermitian=True)
    Pi_4 = cp.Variable((n+1, n+1), hermitian=True)

    # constraints
    constraints = [
        np.identity(n+1) - Pi_1 - Pi_2 - Pi_3 - Pi_4 >> 0,
        Pi_1 >> 0,  # Pi_1 is positive semidefinite
        Pi_2 >> 0,  # Pi_2 is positive semidefinite
        Pi_3 >> 0,  # Pi_3 is positive semidefinite
        Pi_4 >> 0,  # Pi_4 is positive semidefinite
        Pi_1 @ dens_matr_state_2 == 0,
        Pi_1 @ dens_matr_state_3 == 0,
        Pi_1 @ dens_matr_state_4 == 0,
        Pi_2 @ dens_matr_state_1 == 0,
        Pi_2 @ dens_matr_state_3 == 0,
        Pi_2 @ dens_matr_state_4 == 0,
        Pi_3 @ dens_matr_state_1 == 0,
        Pi_3 @ dens_matr_state_2 == 0,
        Pi_3 @ dens_matr_state_4 == 0,
        Pi_4 @ dens_matr_state_1 == 0,
        Pi_4 @ dens_matr_state_2 == 0,
        Pi_4 @ dens_matr_state_3 == 0,
    ]
    if "Pi_1" in zero_operators:
        constraints.append(Pi_1 << 0)
    if "Pi_2" in zero_operators:
        constraints.append(Pi_2 << 0)
    if "Pi_3" in zero_operators:
        constraints.append(Pi_3 << 0)
    if "Pi_4" in zero_operators:
        constraints.append(Pi_4 << 0)

    # objective (p_corr)
    objective = cp.real(cp.trace(dens_matr_state_1 @ Pi_1)*xi_1) + \
        cp.real(cp.trace(dens_matr_state_2 @ Pi_2)*xi_2) + \
        cp.real(cp.trace(dens_matr_state_3 @ Pi_3)*xi_3) + \
            cp.real(cp.trace(dens_matr_state_4 @ Pi_4)*xi_4)

    # solve the problem
    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve(
        solver="SCS", 
        #eps=1e-25, 
        #max_iters=1000,
    )

    return [
        np.trace(np.matmul(Pi_1.value, dens_matr_state_1))*xi_1 \
            + np.trace(np.matmul(Pi_2.value, dens_matr_state_2))*xi_2 \
            + np.trace(np.matmul(Pi_3.value, dens_matr_state_3))*xi_3 \
            + np.trace(np.matmul(Pi_4.value, dens_matr_state_4))*xi_4,  # p_corr
        Pi_1.value,  # Pi_1 obtained from SDP
        Pi_2.value,  # Pi_2 obtained from SDP
        Pi_3.value,  # Pi_3 obtained from SDP
        Pi_4.value  # Pi_4 obtained from SDP
    ]


def sdp_primal_4_states_mixed(
        alpha_1: complex, 
        alpha_2: complex,
        alpha_3: complex, 
        alpha_4: complex,
        xi_1: float,
        xi_2: float,
        xi_3: float,
        xi_4: float,
        n : int,
    ) -> (float, np.array, np.array):

    """
    This function does the SDP for four coherent states. The first state
    corresponding to variable alpha_1 (which is it's eigenvalue) and xi_1
    (it's input probability) is a coherent states. Whereas the remaing three 
    states are treated as one mixed state, which consists of three coherent
    states with eigenvalues alpha_i.

    Parameters
    ----------
        alpha_1 : complex
            eigenvalue of state 1
        alpha_2 : complex
            eigenvalue of state 2
        alpha_3 : complex
            eigenvalue of state 3
        alpha_4 : complex
            eigenvalue of state 4
        xi_1 : float
            input probablity of state 1
        xi_2 : float
            input probablity of state 2
        xi_3 : float
            input probablity of state 3
        xi_4 : float
            input probablity of state 4
        n : int
            The last fock that is represented.
        zero_operators: List[str] = []
            Operators which are forced to be zero matrices. Enter strings as
            "Pi_1" for example.         

    Returns
    -------
        p_corr : float
        Pi_1 : np.array
        Pi_n : np.array
    """

    vec_state_1 = coh_state_vector(alpha_1, n)
    dens_matr_state_1 = density_matrix(vec_state_1)

    vec_state_2 = coh_state_vector(alpha_2, n)
    dens_matr_state_2 = density_matrix(vec_state_2)

    vec_state_3 = coh_state_vector(alpha_3, n)
    dens_matr_state_3 = density_matrix(vec_state_3)

    vec_state_4 = coh_state_vector(alpha_4, n)
    dens_matr_state_4 = density_matrix(vec_state_4)

    dens_matr_state_n = \
        (1/3*dens_matr_state_2) \
        + (1/3*dens_matr_state_3) \
        + (1/3*dens_matr_state_4)

    xi_n = xi_2 + xi_3 + xi_4

    # all probabilities must add up to 1, otherwise error is raised
    if np.round(xi_1 + xi_n, 5) != 1:
        print(xi_1, xi_n)
        print(xi_1 + xi_n)
        raise ValueError

    # Define the variables (POVM elements)
    Pi_1 = cp.Variable((n+1, n+1), hermitian=True)
    Pi_n = cp.Variable((n+1, n+1), hermitian=True)

    constraints = [
        np.identity(n+1) - Pi_1 - Pi_n >> 0,
        Pi_1 >> 0,  # Pi_1 is positive semidefinite
        Pi_n >> 0,  # Pi_2 is positive semidefinite
        dens_matr_state_1 @ Pi_n == 0,
        dens_matr_state_n @ Pi_1 == 0,
    ]

    # objective (p_corr)
    objective = cp.real(cp.trace(dens_matr_state_1 @ Pi_1))*xi_1 \
        + cp.real(cp.trace(dens_matr_state_n @ Pi_n))*xi_n

    # solve the problem
    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve(
        solver="SCS", 
        #eps=1e-25, 
        #max_iters=1000,
    )

    return [
        np.trace(np.matmul(Pi_1.value, dens_matr_state_1))*xi_1 \
            + np.trace(np.matmul(Pi_n.value, dens_matr_state_n))*xi_n,  # p_corr
        #np.trace(np.matmul(Pi_n.value, dens_matr_state_1))*xi_1 \
        #    + np.trace(np.matmul(Pi_1.value, dens_matr_state_n))*xi_n,  # p_err
        Pi_1.value,  # Pi_1 obtained from SDP
        Pi_n.value,  # Pi_2 obtained from SDP
    ]
