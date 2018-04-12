from algorithm_realization.stn_reconstruction_lib import *

pdf,pmf = pdf_generator(mode="pareto")
scatter_wtd(pdf)
scatter_wtd(pmf)