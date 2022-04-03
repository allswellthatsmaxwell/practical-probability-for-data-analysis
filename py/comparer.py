from scipy.stats import beta, norm
import numpy as np, pandas as pd
import plotnine as pn
from typing import Iterable


class Comparer:
    def __init__(self, a: int, b: int, a_name: str, b_name: str, xs: np.ndarray) -> None:
        self.a = a
        self.b = b
        self.n = a + b
        self.mu = self.a / self.n
        self.a_name = a_name
        self.b_name = b_name
        self.xs = xs

    @property
    def beta(self):
        return beta(self.a, self.b)
    
    @property
    def norm(self):
        sd = np.sqrt(self.n * self.mu * (1 - self.mu)) / self.n
        return norm(loc=self.mu, scale=sd)

    @property
    def beta_dat(self):
        return (pd.DataFrame({'x': self.xs, 'p': self.beta.pdf(self.xs)})
                .assign(distribution=f'Beta({self.a}, {self.b})'))

    @property
    def norm_dat(self):
        s = (f'Gaussian({self.mu} * '
             f'sqrt({self.n} * {self.mu} * {(1 - self.mu)})) / {self.n}')
        return (
            pd.DataFrame({'x': self.xs, 'p': self.norm.pdf(self.xs)})
            .assign(distribution=s))

    @property
    def both_dat(self):
        return pd.concat([self.norm_dat, self.beta_dat])

    @property
    def plot(self):
        return (
            pn.ggplot(self.both_dat, pn.aes(x='x', y='p', color='distribution')) +
            pn.geom_line(size=2) +
            pn.labs(x=f'p({self.a_name})', y='density') +
            pn.theme_bw() +
            pn.theme(legend_title=pn.element_blank(),
                     figure_size=(10, 3),
                     legend_position="top",
                     panel_grid=pn.element_blank(),
                     axis_text=pn.element_text(size=14),
                     axis_title=pn.element_text(size=18),
                     legend_text=pn.element_text(size=17)))

    def intervals(self, alpha=0.95):
        return {'beta': abbreviate_items(self.beta.interval(alpha)), 
                'gaussian': abbreviate_items(self.norm.interval(alpha))}

def abbreviate(x: float):
    return float(f"{x:.3f}")

def abbreviate_items(xs: Iterable[float]):
    return [abbreviate(x) for x in xs]