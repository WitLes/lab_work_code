from algorithm_realization.stn_reconstruction_lib import *


pdf = random_sample_recover(mode="gamma")
scatter_wtd(pdf)
pdf2 = pdf_generator(mode="exponential")
scatter_wtd(pdf2)