from scipy.special import iv
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

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

    
def calc_ang_distance(data, rotate_by='stim_angle_brain', exclude_eccen=True):

    # Calculate angular distance between pRF and stimulus
    data['ang_dist'] = data['full-angle'] - data[rotate_by]

    # Sets degree range to [-180, 180]
    fix_deg = lambda x: (x + 180) % 360 - 180
    data['ang_dist'] = data['ang_dist'].apply(fix_deg)

    # Restrict to nearby eccentricity
    if exclude_eccen:
        f = lambda x: abs(x['full-eccen'] - x['stim_eccen']) > x['full-sigma']
        far_eccen = data.apply(f, axis=1)
        data.loc[far_eccen, 'ang_dist'] = np.nan

    # Bin these distances
    dist_bins = np.arange(-180, 220, 20) - 10
    bins = pd.cut(data['ang_dist'], bins=dist_bins)
    bins = bins.apply(center_bin).astype(float)
    bins[bins == -180.0] = 180.0 # collapse -180 and 180 bins
    data['ang_dist_bin'] = bins

    return data


def calc_eccen_bin(data, exclude_angle=True):

    # Bin according to eccentricity
    edist_bins = np.arange(.5, 8.5, .5)
    bins = pd.cut(data['full-eccen'], bins=edist_bins)
    bins = bins.apply(center_bin).astype(float)
    data['eccen_bin'] = bins

    # Restrict to nearby polar angle
    if exclude_angle:
        f = lambda x: abs(x['full-angle'] - x['stim_angle_brain']) > 15
        far_polar = data.apply(f, axis=1)
        data.loc[far_polar, 'eccen_bin'] = np.nan

    return data


def center_bin(x):
    center = (x.left.astype(float) + x.right.astype(float))/2
    return center


def vonmises(theta, loc, kappa, scale):
    p = scale * np.exp(kappa*np.cos(theta-loc))/(2*np.pi*iv(0,kappa))
    return p


def diff_vonmises(theta, loc, kappa1, scale1, kappa2, scale2):
    p1 = vonmises(theta, loc, kappa1, scale1)
    p2 = vonmises(theta, loc, kappa2, scale2)
    return (p1 - p2)


def fwhm(X, Y):
    d = Y - (max(Y) / 2)
    indexes = np.where(d > 0)[0]
    return abs(X[indexes[-1]] - X[indexes[0]])


def norm_group(data, yvar='beta', xvar='ang_dist_bin', group_cols=[]):

    # Take median of observations within each data group and distance bin
    group_cols = ['subj', 'hemi', 'roi', 'task'] + group_cols
    data = data.groupby(group_cols + [xvar]).median()[yvar].reset_index()

    # Divide each subj response by norm
    norm_data = []
    for (cols, g) in data.groupby(group_cols):
        sd = g.copy()
        sd.loc[:, 'norm'] = np.linalg.norm(sd[yvar])
        sd.loc[:, yvar+'_norm'] = sd[yvar] / sd['norm']
        norm_data.append(sd)
    norm_data = pd.concat(norm_data)

    # Average across subjects and multiply by average norm to get units back
    norm_data = norm_data.groupby(group_cols[1:] + [xvar]).mean().reset_index()
    norm_data[yvar+'_adj'] = norm_data[yvar+'_norm']*norm_data['norm']

    norm_data = norm_data.drop(columns=['norm', yvar+'_norm'])

    return norm_data


def fit_diff_vonmises(data, yvar, xvar='ang_dist_bin', group_cols=[], drop_cols=[]):

    # Convert binned data to radians for fitting
    data[xvar+'_rad'] = data[xvar].apply(np.deg2rad)

    # Highly sampled x range in radians
    x = np.deg2rad(np.arange(-180, 180, 1))

    # Loop over each roi and model and fit von mises to group data
    params = []
    assumed_cols = list(filter(lambda x: x not in drop_cols, ['hemi', 'roi', 'task']))

    for cols, g in data.groupby(assumed_cols+group_cols):

        try:

            # Fit diff of von mises to binned data
            bounds = [[-np.pi, 0, 0, 0, 0],[np.pi, np.inf, np.inf, np.inf, np.inf]]
            xbin = g[xvar+'_rad'].values
            ybin = g[yvar].values
            popt, pcov = curve_fit(diff_vonmises, xbin, ybin, bounds=bounds, maxfev=10000)

            # Get yhat on highly sampled x and von mises parameters
            yhat = diff_vonmises(x, *popt)
            loc, kappa1, scale1, kappa2, scale2 = popt

            # Calculate fwhm
            width = fwhm(x, yhat)

            # Save all info for this fit
            p = dict(func='diff_vonmises',
                     loc=loc, loc_deg=np.rad2deg(loc),
                     kappa1=kappa1, scale1=scale1,
                     kappa2=kappa2, scale2=scale2,
                     maxr=max(yhat), minr=min(yhat),
                     amp=max(yhat)-min(yhat),
                     fwhm=width, fwhm_deg=np.rad2deg(width),
                     r2=r2_score(ybin, diff_vonmises(xbin, *popt))
                     )
        except:
            p = dict()

        group_df = g.reset_index()[assumed_cols + group_cols].iloc[:1]
        p = pd.DataFrame(p, index=[0])
        p = pd.concat([group_df, p], axis=1)
        params.append(p)

    params = pd.concat(params, sort=False).reset_index(drop=True)

    return params
