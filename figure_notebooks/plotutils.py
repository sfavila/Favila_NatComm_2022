from scipy.special import iv
from sklearn import linear_model
import numpy as np
import seaborn as sns
import matplotlib as mpl

def default_data():
    subjects = ['wlsubj004',
                'wlsubj007',
                'wlsubj023',
                'wlsubj045',
                'wlsubj061',
                'wlsubj064',
                'wlsubj065',
                'wlsubj067',
                'wlsubj070']
    rois = ['V1','V2', 'V3', 'hV4', 'LO', 'V3ab']
    tasks = ['perception', 'memory']

    return subjects, rois, tasks


def set_plot_style():
    sns.set_context('notebook', font_scale=1.5)
    mpl.rcParams['font.sans-serif'] = ['Arial']


def vonmises(theta, loc, kappa, scale):
    p = scale * np.exp(kappa*np.cos(theta-loc))/(2*np.pi*iv(0,kappa))
    return p


def diff_vonmises(theta, loc, kappa1, scale1, kappa2, scale2):
    p1 = vonmises(theta, loc, kappa1, scale1)
    p2 = vonmises(theta, loc, kappa2, scale2)
    return (p1 - p2)


def pred_scatter_bootstrap(boot_scatter, xgrid, yvar):

    betas, yhats = [], []
    for i, b in boot_scatter.groupby("n_boot"):
        x = b[yvar+'_pred'].values.reshape(-1, 1)
        y = b[yvar].values.reshape(-1, 1)
        regr = linear_model.LinearRegression()
        regr.fit(x, np.nan_to_num(y))
        betas.append(regr.coef_.flatten())
        yhats.append(regr.predict(xgrid).T)

    betas_95 = np.nanpercentile(betas, [2.5, 97.5])

    yhats = np.vstack(yhats)
    err_bands_68 = np.nanpercentile(yhats, [16, 84], axis=0)
    err_bands_95 = np.nanpercentile(yhats, [2.5, 97.5], axis=0)

    return betas_95, err_bands_68, err_bands_95
