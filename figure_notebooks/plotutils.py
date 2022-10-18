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


def params_ci(params_boot):

    # Melt dataframe
    pboots = params_boot.melt(id_vars=['hemi', 'roi', 'task', 'func', 'n_boot'],
                              value_vars= ['loc_deg', 'amp', 'fwhm_deg'],
                              var_name='param')

    # Calculate confidence intervals
    g = pboots.groupby(['hemi', 'roi', 'task', 'param'])['value']
    conf_68 = g.apply(np.nanpercentile, [16, 84]).reset_index(name=68)
    conf_95 = g.apply(np.nanpercentile, [2.5, 97.5]) .reset_index(name=95)
    params_conf_wide = conf_68.merge(conf_95)
    params_conf = params_conf_wide.melt(id_vars=['hemi', 'roi', 'task', 'param'],
                                        var_name='conf_level', value_name='ci')

    return params_conf


def draw_lm_ci(g, boot_scatter, yvar, pal_pred):

    betas_95 = {'perception':dict(), 'memory':dict()}

    fd = [d for (i, d) in g.facet_data() if not d.empty]
    for i, d in enumerate(fd):
        ax = g.axes.flatten()[i]
        xmin = ax.get_lines()[0].get_data()[0].min()
        xmax = ax.get_lines()[0].get_data()[0].max()
        x_grid = np.linspace(xmin, xmax, 100).reshape(-1, 1)

        t,m = d[['task', 'prf_model']].iloc[0]
        bs = boot_scatter.query("task==@t & prf_model==@m")
        b_95, err_68, err_95 = pred_scatter_bootstrap(bs, x_grid, yvar)
        betas_95[t][m] = b_95

        ax.fill_between(x_grid.squeeze(), *err_68, facecolor=pal_pred[m], alpha=.4)
        ax.plot(x_grid.squeeze(), err_95[0], color=pal_pred[m], lw=1.5, alpha=.4)
        ax.plot(x_grid.squeeze(), err_95[1], color=pal_pred[m], lw=1.5, alpha=.4)

    return g, betas_95


def pred_scatter_bootstrap(boot_scatter, xgrid, yvar):

    # Fit linear model for each boot
    betas, yhats = [], []
    for i, b in boot_scatter.groupby("n_boot"):
        x = b[yvar+'_pred'].values.reshape(-1, 1)
        y = b[yvar].values.reshape(-1, 1)
        regr = linear_model.LinearRegression()
        regr.fit(x, np.nan_to_num(y))
        betas.append(regr.coef_.flatten())
        yhats.append(regr.predict(xgrid).T)

    # Beta confidence intervals
    betas_95 = np.nanpercentile(betas, [2.5, 97.5])

    # Yhat confidence intervals
    yhats = np.vstack(yhats)
    err_bands_68 = np.nanpercentile(yhats, [16, 84], axis=0)
    err_bands_95 = np.nanpercentile(yhats, [2.5, 97.5], axis=0)

    return betas_95, err_bands_68, err_bands_95
