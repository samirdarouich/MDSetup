import numpy as np


def compute_VRH(stiffness_tensor: np.ndarray):
    """
    Compute the Voigt-Reuss-Hill (VRH) averages of the elastic constants.

    Parameters:
        stiffness_tensor (np.ndarray): The 6x6 stiffness tensor representing the elastic constants.

    Returns:
        tuple: A tuple containing the VRH averages of the elastic constants in the following order:
            - K (float): Bulk modulus.
            - G (float): Shear modulus.
            - E (float): Young's modulus.
            - nu (float): Poisson's ratio.

    Notes:
        - The stiffness tensor should be in Voigt notation.
        - The stiffness tensor should be symmetric.
        - The stiffness tensor should be positive definite.

    References:
        - Voigt, W. (1928). Lehrbuch der Kristallphysik. Teubner.
        - Reuss, A. (1929). Berechnung der Fließgrenze von Mischkristallen auf Grund der Plastizitätsbedingung für Einkristalle. ZAMM - Journal of Applied Mathematics and Mechanics / Zeitschrift für Angewandte Mathematik und Mechanik, 9(1), 49–58.
        - Hill, R. (1952). The Elastic Behaviour of a Crystalline Aggregate. Proceedings of the Physical Society. Section A, 65(5), 349–354.
    """
    c_sq = (stiffness_tensor[0][0] + stiffness_tensor[0][1]) * stiffness_tensor[2][
        2
    ] - 2 * stiffness_tensor[0][2] ** 2
    M = (
        stiffness_tensor[0][0]
        + stiffness_tensor[0][1]
        + 2 * stiffness_tensor[2][2]
        - 4 * stiffness_tensor[0][2]
    )

    K_V = 1 / 9 * (
        stiffness_tensor[0][0] + stiffness_tensor[1][1] + stiffness_tensor[2][2]
    ) + 2 / 9 * (
        stiffness_tensor[0][1] + stiffness_tensor[0][2] + stiffness_tensor[1][2]
    )

    K_R = c_sq / M

    G_V = 1 / 30 * (M + 12 * stiffness_tensor[3][3] + 12 * stiffness_tensor[5][5])

    G_R = (
        5
        / 2
        * c_sq
        * stiffness_tensor[3][3]
        * stiffness_tensor[5][5]
        / (
            c_sq * (stiffness_tensor[3][3] + stiffness_tensor[5][5])
            + 3 * K_V * stiffness_tensor[3][3] * stiffness_tensor[5][5]
        )
    )

    K = 0.5 * (K_V + K_R)
    G = 0.5 * (G_V + G_R)
    E = 9 * K * G / (3 * K + G)
    nu = (3 * K - 2 * G) / (6 * K + 2 * G)

    return K, G, E, nu
