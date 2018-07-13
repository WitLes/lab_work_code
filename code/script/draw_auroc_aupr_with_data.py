import numpy as np
import matplotlib.pyplot as plt


# aupr
er_gaussian_aupr = [0.2996,0.6774,0.9332,0.9776,0.9914,0.9979,0.9983,0.9996,0.9998,0.9996]
er_uniform_aupr = [0.2211,0.4389,0.659,0.8377,0.9158,0.9676,0.9873,0.9941,0.9963,0.9989]
er_gumbel_aupr = [0.321667519,0.682377408,0.888793596,0.945572223,0.977980137,0.988729147,0.992263356,0.995381894,0.996421558,0.997631125]
ba_gaussian_aupr = [0.216325749,0.338296157,0.416844617,0.612134709,0.686212235,0.755759651,0.802410178,0.836799429,0.887608268,0.921491036]
ba_uniform_aupr = [0.14637228,0.25487676,0.360836694,0.52657761,0.672149892,0.780455787,0.870736956,0.916936094,0.951164162,0.962470962]
ba_gumbel_aupr = [0.177512269,0.305002648,0.448996454,0.569307718,0.662552246,0.734044541,0.77950873,0.816072739,0.864148038,0.892991378]
ws_gaussian_aupr = [0.959562933,0.996717061,0.999030719,0.999883106,0.999888591,0.999959443,0.999934987,1,0.999921725,1]
ws_uniform_aupr = [0.977799012,0.995034883,0.997777328,0.999667525,0.99987642,0.999814964,0.999922063,0.99997604,0.99994523,1]
ws_gumbel_aupr = [0.960489429,0.99309057,0.99516679,0.996229077,0.997947982,0.997915074,0.997545827,0.998225378,0.997801121,0.997698777]

football_gaussian_aupr=[0.415166985,0.721717099,0.874085604,0.930506571,0.965107239,0.97867962,0.981414252,0.988004961,0.989497359,0.994363946]
football_uniform_aupr=[0.459526564,0.684386926,0.775730936,0.862285205,0.900320947,0.936533606,0.942280259,0.960246452,0.966164958,0.972243471]
football_gumbel_aupr=[0.454027395,0.75117225,0.864646345,0.927492555,0.944639977,0.966492244,0.976357516,0.978630909,0.984149225,0.988427944]
lattice2d_gaussian_aupr=[0.854984419,0.964494756,0.996809439,0.99412212,1,1,1,1,1,1]
lattice2d_uniform_aupr=[0.849998818,0.966090842,0.996401044,0.999080381,1,0.999708284,1,1,1,1]
lattice2d_gumbel_aupr=[0.870614883,0.993189276,0.997339222,0.999221609,0.998583653,0.99993727,0.999965549,0.999848146,0.999972341,0.99996228]
sierpinski_gaussian_aupr=[0.977198254,0.99496041,0.999120523,0.999734483,0.999898533,0.999950555,0.99997828,0.999997892,0.999975241,0.999989714]
sierpinski_uniform_aupr=[0.965180694,0.993926459,0.999111647,0.999670618,1,1,1,1,1,1]
sierpinski_gumbel_aupr=[0.975939787,0.995319066,0.996279406,0.998256457,0.998834894,0.999231706,0.999375592,0.99947123,0.999602403,0.99938196]
# auroc
er_gaussian_auroc = [0.9044,0.9858,0.9976,0.9993,0.9998,0.9999,1,1,1,1]
er_uniform_auroc = [0.8548,0.9505,0.9816,0.9936,0.9971,0.999,0.9996,0.9998,0.9999,1]
er_gumbel_auroc = [0.922862247,0.981896903,0.995051064,0.997510433,0.998983065,0.999424418,0.999613517,0.999731024,0.999790415,0.99986646]
ba_gaussian_auroc = [0.887609996,0.961722135,0.981653306,0.988954299,0.993513485,0.995867521,0.991638296,0.99766042,0.998611001,0.998631397]
ba_uniform_auroc = [0.789394227,0.91082604,0.959474829,0.981600316,0.990246998,0.994240971,0.997109021,0.998299753,0.999084896,0.999336892]
ba_gumbel_auroc = [0.846952195,0.959474829,0.967847587,0.979255514,0.98660568,0.990179701,0.99278263,0.994068346,0.995732322,0.996853941]
ws_gaussian_auroc = [0.999207873,0.999923347,0.999975749,0.999996458,0.999996785,0.999998754,0.999998232,1,0.999997991,1]
ws_uniform_auroc = [0.99941383,0.999876318,0.999946566,0.999990836,0.999996149,0.999994853,0.999998022,0.999999254,0.999998294,1]
ws_gumbel_auroc = [0.999133518,0.999807812,0.999867745,0.99988533,0.999936556,0.999933921,0.999919997,0.999942647,0.999928892,0.999924725]

football_gaussian_auroc=[0.88529204,0.971445019,0.989391701,0.994823403,0.997494515,0.998644523,0.998963529,0.999404967,0.999487515,0.99963703]
football_uniform_auroc=[0.887614192,0.954677165,0.9739372,0.98666889,0.990845635,0.994367406,0.995240835,0.996826917,0.997457545,0.997965806]
football_gumbel_auroc=[0.903304335,0.971599838,0.98671128,0.993769718,0.995274531,0.997435962,0.998065429,0.998448125,0.998713583,0.999065441]
lattice2d_gaussian_auroc=[0.997024039,0.999359294,0.99998384,0.999982035,1,1,1,1,1,1]
lattice2d_uniform_auroc=[0.996011647,0.999366271,0.999948672,0.999973806,1,0.999991177,1,1,1,1]
lattice2d_gumbel_auroc=[0.997138539,0.999908811,0.999957786,0.999977772,0.99997083,0.999997577,0.999998672,0.999993699,0.99999894,0.999998515]
sierpinski_gaussian_auroc=[0.99941831,0.999858903,0.999971862,0.999991293,0.999996613,0.999998356,0.999999269,0.999999929,0.999999167,0.999999654]
sierpinski_uniform_auroc=[0.999083146,0.999854831,0.99997767,0.999990568,1,1,1,1,1,1]
sierpinski_gumbel_auroc=[0.999358257,0.99985101,0.99987876,0.999940808,0.999960642,0.999973625,0.99997897,0.999982074,0.999986336,0.999978245]

# stochastic network

all_font_size = 16
axis_font_size = 13
legend_font_size = 13


# ER
X_axis = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
a= plt.figure(figsize=(14,8))
plt.subplot(2, 3, 1)
plt.xlabel("C=M/N",fontsize=all_font_size)
plt.ylabel("AUROC",fontsize=all_font_size)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.xticks(fontsize=axis_font_size)
plt.yticks(fontsize=axis_font_size)
plt.plot(X_axis,er_uniform_auroc,"bs-",markersize=7,label="(ER,Uniform)",linewidth=1)
plt.plot(X_axis,er_gaussian_auroc,"ro-",markersize=7,label="(ER,Gaussian)",linewidth=1)
plt.plot(X_axis,er_gumbel_auroc,"g^",markersize=7,label="(ER,Gumbel)",linewidth=1)
plt.legend(fontsize=legend_font_size)

plt.subplot(2,3, 4)
plt.xlabel("C=M/N",fontsize=all_font_size)
plt.ylabel("AUPR",fontsize=all_font_size)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.xticks(fontsize=axis_font_size)
plt.yticks(fontsize=axis_font_size)
plt.plot(X_axis,er_uniform_aupr,"bs-",markersize=7,label="(ER,Uniform)",linewidth=1)
plt.plot(X_axis,er_gaussian_aupr,"ro-",markersize=7,label="(ER,Gaussian)",linewidth=1)
plt.plot(X_axis,er_gumbel_aupr,"g^-",markersize=7,label="(ER,Gumbel)",linewidth=1)
plt.legend(fontsize=legend_font_size)

plt.subplot(2,3, 2)
plt.xlabel("C=M/N",fontsize=all_font_size)
plt.ylabel("AUROC",fontsize=all_font_size)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.xticks(fontsize=axis_font_size)
plt.yticks(fontsize=axis_font_size)
plt.plot(X_axis,ba_uniform_auroc,"bs-",markersize=7,label="(BA,Uniform)",linewidth=1)
plt.plot(X_axis,ba_gaussian_auroc,"ro-",markersize=7,label="(BA,Gaussian)",linewidth=1)
plt.plot(X_axis,ba_gumbel_auroc,"g^-",markersize=7,label="(BA,Gumbel)",linewidth=1)
plt.legend(fontsize=legend_font_size)

plt.subplot(2,3, 5)
plt.xlabel("C=M/N",fontsize=all_font_size)
plt.ylabel("AUPR",fontsize=all_font_size)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.xticks(fontsize=axis_font_size)
plt.yticks(fontsize=axis_font_size)
plt.plot(X_axis,ba_uniform_aupr,"bs-",markersize=7,label="(BA,Uniform)",linewidth=1)
plt.plot(X_axis,ba_gaussian_aupr,"ro-",markersize=7,label="(BA,Gaussian)",linewidth=1)
plt.plot(X_axis,ba_gumbel_aupr,"g^-",markersize=7,label="(BA,Gumbel)",linewidth=1)
plt.legend(fontsize=legend_font_size,loc="lower right")

plt.subplot(2,3,3)
plt.xlabel("C=M/N",fontsize=all_font_size)
plt.ylabel("AUROC",fontsize=all_font_size)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.xticks(fontsize=axis_font_size)
plt.yticks(fontsize=axis_font_size)
plt.plot(X_axis,ws_uniform_auroc,"bs-",markersize=7,label="(WS,Uniform)",linewidth=1)
plt.plot(X_axis,ws_gaussian_auroc,"ro-",markersize=7,label="(WS,Gaussian)",linewidth=1)
plt.plot(X_axis,ws_gumbel_auroc,"g^-",markersize=7,label="(WS,Gumbel)",linewidth=1)
plt.legend(fontsize=legend_font_size)

plt.subplot(2,3, 6)
plt.xlabel("C=M/N",fontsize=all_font_size)
plt.ylabel("AUPR",fontsize=all_font_size)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.xticks(fontsize=axis_font_size)
plt.yticks(fontsize=axis_font_size)
plt.plot(X_axis,ws_uniform_aupr,"bs-",markersize=7,label="(WS,Uniform)",linewidth=1)
plt.plot(X_axis,ws_gaussian_aupr,"ro-",markersize=7,label="(WS,Gaussian)",linewidth=1)
plt.plot(X_axis,ws_gumbel_aupr,"g^-",markersize=7,label="(WS,Gumbel)",linewidth=1)
plt.legend(fontsize=legend_font_size,loc="lower right")
plt.subplots_adjust(top=0.99,bottom=0.07,left=0.06,right=0.98,wspace=0.27,hspace=0.23)
plt.savefig("../figure/stochastic_auroc_aupr.svg", format="svg")
plt.savefig("../figure/stochastic_auroc_aupr.png", dpi=600, format="png")
plt.savefig("../figure/stochastic_auroc_aupr.eps", format="eps")
plt.show()

"""
# fixed network
X_axis = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
# ER

a= plt.figure(figsize=(14,8))
plt.subplot(2, 3, 1)
plt.xlabel("C=M/N",fontsize=all_font_size)
plt.ylabel("AUROC",fontsize=all_font_size)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.xticks(fontsize=axis_font_size)
plt.yticks(fontsize=axis_font_size)
plt.plot(X_axis,football_uniform_auroc,"bs-",markersize=7,label="(FT,Uniform)",linewidth=1)
plt.plot(X_axis,football_gaussian_auroc,"ro-",markersize=7,label="(FT,Gaussian)",linewidth=1)
plt.plot(X_axis,football_gumbel_auroc,"g^",markersize=7,label="(FT,Gumbel)",linewidth=1)
plt.legend(fontsize=legend_font_size)

plt.subplot(2,3, 4)
plt.xlabel("C=M/N",fontsize=all_font_size)
plt.ylabel("AUPR",fontsize=all_font_size)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.xticks(fontsize=axis_font_size)
plt.yticks(fontsize=axis_font_size)
plt.plot(X_axis,football_uniform_aupr,"bs-",markersize=7,label="(FT,Uniform)",linewidth=1)
plt.plot(X_axis,football_gaussian_aupr,"ro-",markersize=7,label="(FT,Gaussian)",linewidth=1)
plt.plot(X_axis,football_gumbel_aupr,"g^-",markersize=7,label="(FT,Gumbel)",linewidth=1)
plt.legend(fontsize=legend_font_size)

plt.subplot(2,3, 2)
plt.xlabel("C=M/N",fontsize=all_font_size)
plt.ylabel("AUROC",fontsize=all_font_size)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.xticks(fontsize=axis_font_size)
plt.yticks(fontsize=axis_font_size)
plt.plot(X_axis,lattice2d_uniform_auroc,"bs-",markersize=7,label="(LT,Uniform)",linewidth=1)
plt.plot(X_axis,lattice2d_gaussian_auroc,"ro-",markersize=7,label="(LT,Gaussian)",linewidth=1)
plt.plot(X_axis,lattice2d_gumbel_auroc,"g^-",markersize=7,label="(LT,Gumbel)",linewidth=1)
plt.legend(fontsize=legend_font_size)

plt.subplot(2,3, 5)
plt.xlabel("C=M/N",fontsize=all_font_size)
plt.ylabel("AUPR",fontsize=all_font_size)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.xticks(fontsize=axis_font_size)
plt.yticks(fontsize=axis_font_size)
plt.plot(X_axis,lattice2d_uniform_aupr,"bs-",markersize=7,label="(LT,Uniform)",linewidth=1)
plt.plot(X_axis,lattice2d_gaussian_aupr,"ro-",markersize=7,label="(LT,Gaussian)",linewidth=1)
plt.plot(X_axis,lattice2d_gumbel_aupr,"g^-",markersize=7,label="(LT,Gumbel)",linewidth=1)
plt.legend(fontsize=legend_font_size)

plt.subplot(2,3,3)
plt.xlabel("C=M/N",fontsize=all_font_size)
plt.ylabel("AUROC",fontsize=all_font_size)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.xticks(fontsize=axis_font_size)
plt.yticks(fontsize=axis_font_size)
plt.plot(X_axis,sierpinski_uniform_auroc,"bs-",markersize=7,label="(SP,Uniform)",linewidth=1)
plt.plot(X_axis,sierpinski_gaussian_auroc,"ro-",markersize=7,label="(SP,Gaussian)",linewidth=1)
plt.plot(X_axis,sierpinski_gumbel_auroc,"g^-",markersize=7,label="(SP,Gumbel)",linewidth=1)
plt.legend(fontsize=legend_font_size)

plt.subplot(2,3, 6)
plt.xlabel("C=M/N",fontsize=all_font_size)
plt.ylabel("AUPR",fontsize=all_font_size)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.xticks(fontsize=axis_font_size)
plt.yticks(fontsize=axis_font_size)
plt.plot(X_axis,sierpinski_uniform_aupr,"bs-",markersize=7,label="(SP,Uniform)",linewidth=1)
plt.plot(X_axis,sierpinski_gaussian_aupr,"ro-",markersize=7,label="(SP,Gaussian)",linewidth=1)
plt.plot(X_axis,sierpinski_gumbel_aupr,"g^-",markersize=7,label="(SP,Gumbel)",linewidth=1)
plt.legend(fontsize=legend_font_size)
plt.subplots_adjust(top=0.99,bottom=0.07,left=0.06,right=0.98,wspace=0.27,hspace=0.23)
plt.savefig("../figure/fixed_auroc_aupr.svg", format="svg")
plt.savefig("../figure/fixed_auroc_aupr.png", dpi=600, format="png")
plt.savefig("../figure/fixed_auroc_aupr.eps", format="eps")
plt.show()
"""