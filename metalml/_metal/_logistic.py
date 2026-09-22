"""GPU IRLS training for binary logistic regression.

The expensive part of a Newton iteration is the weighted Gram matrix
``X'WX`` and the score ``X'(p-y)``; both are ``O(n d^2)`` and run on Metal as
ordinary matrix products. Only the tiny ``d x d`` positive-definite solve
stays on the CPU, mirroring the Ridge estimator's design.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from ._session import Session


class LogisticFallback(Exception):
    """The GPU Newton iteration cannot safely finish; callers use sklearn."""


class LogisticIRLS(Session):
    def __init__(self, runtime, x, y, alpha, fit_intercept, max_iter, tol):
        super().__init__(runtime)
        self.x, self.y = x, y
        self.alpha, self.fit_intercept = alpha, fit_intercept
        self.max_iter, self.tol = max_iter, tol
        self.iteration = 0

    def __enter__(self):
        super().__enter__()
        try:
            n, d = self.x.shape
            D = d + int(self.fit_intercept)
            if self.fit_intercept:
                # A constant column lets the intercept share the Gram product;
                # its diagonal is left unpenalized below.
                xa = np.empty((n, D), np.float32)
                xa[:, :d] = self.x
                xa[:, d] = 1.0
            else:
                xa = np.ascontiguousarray(self.x, np.float32)
            self.D = D
            self.xa = self.upload(xa)
            self.yb = self.upload(self.y)
            self.beta = self.empty((D, 1))
            self.eta = self.empty((n, 1))
            self.root = self.empty((n,))
            self.resid = self.empty((n, 1))
            self.xw = self.empty((n, D))
            self.H = self.empty((D, D))
            self.g = self.empty((D, 1))
            self.beta.array.fill(0.0)
            return self
        except BaseException:
            import sys

            super().__exit__(*sys.exc_info())
            raise

    def _newton_step(self):
        n, D = self.xa.shape[0], self.D
        command = self.command()
        self.matmul(command, self.xa, self.beta, self.eta)
        self.encode(
            command, "logistic_deriv", [self.eta, self.yb, self.root, self.resid], [n], (n,)
        )
        self.encode(command, "scale_rows", [self.xa, self.root, self.xw], [n, D], (n * D,))
        self.matmul(command, self.xw, self.xw, self.H, transpose_left=True)
        self.matmul(command, self.xa, self.resid, self.g, transpose_left=True)
        self.submit(command, dispatches=5)

    def run(self):
        D = self.D
        penalty = np.full(D, float(self.alpha))
        if self.fit_intercept:
            penalty[-1] = 0.0
        beta = np.zeros(D)
        iteration = 0
        while iteration < self.max_iter:
            iteration += 1
            self._newton_step()
            gram = self.H.read().astype(np.float64)
            score = self.g.read().astype(np.float64).reshape(-1)
            if not (np.isfinite(gram).all() and np.isfinite(score).all()):
                raise LogisticFallback("Non-finite sufficient statistics")
            system = gram + np.diag(penalty)
            rhs = -score - penalty * beta
            try:
                delta = cho_solve(cho_factor(system, check_finite=False), rhs, check_finite=False)
            except np.linalg.LinAlgError as exc:
                raise LogisticFallback("Hessian is not positive definite") from exc
            if not np.isfinite(delta).all():
                raise LogisticFallback("Non-finite Newton step")
            beta = beta + delta
            self.beta.array[:, 0] = beta.astype(np.float32)
            if np.max(np.abs(delta)) < self.tol:
                break
        self.iteration = iteration
        coefficients = beta[: -1] if self.fit_intercept else beta
        intercept = beta[-1] if self.fit_intercept else 0.0
        return coefficients.astype(np.float32), float(intercept), self.iteration
