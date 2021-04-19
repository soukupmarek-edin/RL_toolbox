import numpy as np
from scipy import stats, integrate


class Distribution:

    def __init__(self, dist, N):
        self.N = N
        self.dist = dist

    def pdf(self, x):
        return self.dist.pdf(x)

    def cdf(self, x):
        return self.dist.cdf(x)

    def fos_pdf(self, x):
        N, cdf, pdf = self.N, self.dist.cdf, self.dist.pdf
        return N * cdf(x) ** (N - 1) * pdf(x)

    def fos_cdf(self, x):
        N, cdf, pdf = self.N, self.dist.cdf, self.dist.pdf
        return cdf(x) ** N

    def sos_pdf(self, x):
        N, cdf, pdf = self.N, self.dist.cdf, self.dist.pdf
        return N * (N - 1) * (1 - cdf(x)) * cdf(x) ** (N - 2) * pdf(x)

    def sos_cdf(self, x):
        N, cdf, pdf = self.N, self.dist.cdf, self.dist.pdf
        return N * cdf(x) ** (N - 1) - (N - 1) * cdf(x) ** N

    def exp(self):
        return integrate.quad(lambda x: x * self.pdf(x), -np.inf, np.inf)[0]

    def exp_fos(self):
        return integrate.quad(lambda x: x * self.fos_pdf(x), -np.inf, np.inf)[0]

    def exp_sos(self):
        return integrate.quad(lambda x: x * self.sos_pdf(x), -np.inf, np.inf)[0]