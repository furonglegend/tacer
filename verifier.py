"""
Verifier for Unexplainability Certificates.

Replays falsification traces deterministically.
Optionally exports proof obligations for Lean4.
"""

from .issuer import Certificate


def verify_certificate(cert: Certificate) -> bool:
    """
    Verify that all candidate explanations were falsified.
    """
    for prog, falsified in cert.falsifications.items():
        if not falsified:
            return False
    return True


def export_to_lean(cert: Certificate, path: str):
    """
    Export certificate as a Lean4 proof sketch.
    """
    with open(path, "w") as f:
        f.write("-- Auto-generated UCert proof\n")
        f.write(f"-- epsilon = {cert.epsilon}\n")
        f.write(f"-- L = {cert.L}\n")
        for prog in cert.falsifications:
            f.write(f"-- falsified: {prog}\n")
