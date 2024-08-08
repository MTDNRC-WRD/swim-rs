import os
import time

import numpy as np
import pandas as pd

from model.etd import obs_field_cycle
from swim.config import ProjectConfig
from swim.input import SamplePlots

import matplotlib.pyplot as plt

from scipy.optimize import minimize, Bounds
import random

from prep import info

def optimize_fields(ini_path, debug_flag=False):
    start_time = time.time()

    proj_dir = os.path.dirname(ini_path)

    config = ProjectConfig()
    config.read_config(ini_path)

    fields = SamplePlots()
    fields.initialize_plot_data(config)

    df = obs_field_cycle.field_day_loop(config, fields, debug_flag=debug_flag)

    # debug_flag=False just returns the ndarray for writing
    if not debug_flag:
        eta_result, swe_result = df
        for i, fid in enumerate(fields.input['order']):
            pred_eta, pred_swe = eta_result[:, i], swe_result[:, i]
            np.savetxt(os.path.join(d, 'pest', 'pred', 'pred_eta_{}.np'.format(fid)), pred_eta)
            np.savetxt(os.path.join(d, 'pest', 'pred', 'pred_swe_{}.np'.format(fid)), pred_swe)
        end_time = time.time()
        print('\n\nExecution time: {:.2f} seconds'.format(end_time - start_time))

    # debug returns a dataframe
    if debug_flag:

        targets = fields.input['order']
        first = True

        print('Warning: model runner is set to debug=True, it will not write results accessible to PEST++')

        for i, fid in enumerate(targets):

            pred_et = df[fid]['et_act'].values

            obs_et = os.path.join(proj_dir, 'obs/obs_eta_{}.np'.format(fid))
            obs_et = np.loadtxt(obs_et)
            cols = ['et_obs'] + list(df[fid].columns)
            df[fid]['et_obs'] = obs_et
            df[fid] = df[fid][cols]
            sdf = df[fid].loc['2017-01-01': '2017-12-31']

            comp = pd.DataFrame(data=np.vstack([obs_et, pred_et]).T, columns=['obs', 'pred'], index=df[fid].index)
            comp['eq'] = comp['obs'] == comp['pred']
            comp['capture'] = df[fid]['capture']

            rmse = np.sqrt(((pred_et - obs_et) ** 2).mean())
            end_time = time.time()

            if first:
                print('Execution time: {:.2f} seconds'.format(end_time - start_time))
                first = False

            print('{}: Mean Obs: {:.2f}, Mean Pred: {:.2f}'.format(fid, obs_et.mean(), pred_et.mean()))
            print('{}: RMSE: {:.4f}'.format(fid, rmse))

            comp = comp.loc[sdf[sdf['capture'] == 1.0].index]
            pred_et, obs_et = comp['pred'], comp['obs']
            rmse = np.sqrt(((pred_et - obs_et) ** 2).mean())
            print('{}: RMSE Capture Dates: {:.4f}'.format(fid, rmse))

            obs_swe = os.path.join(proj_dir, 'obs/obs_swe_{}.np'.format(fid))
            obs_swe = np.loadtxt(obs_swe)
            cols = ['swe_obs'] + list(df[fid].columns)
            df[fid]['swe_obs'] = obs_swe
            df[fid] = df[fid][cols]
            swe_df = df[fid].loc['2010-01-01': '2021-01-01'][['swe_obs', 'swe']]
            swe_df.dropna(axis=0, inplace=True)
            pred_swe = swe_df['swe'].values
            obs_swe = swe_df['swe_obs'].values
            rmse = np.sqrt(((pred_swe - obs_swe) ** 2).mean())
            print('{}: RMSE SWE: {:.4f}\n\n\n\n'.format(fid, rmse))


def run_fields(ini_path, debug_flag=False):
    """ Slight variation on optimize_fields above, to be used without PEST++.
    Main differences:
    - This function passes a dictionary of parameters based off of params.csv to run the model,
    instead of using the single-parameter files found in the pest/mult directory (specified as 'calibration_folder' in
    the .toml control file).
    - Lots of printing is suppressed here. Additional printing present in model/etd/obs_field_cycle line 85/86.
    - Returns rmse values as arrays, to be fed into sensitivity analysis functions.
    """

    start_time = time.time()

    config = ProjectConfig()
    config.read_config(ini_path)

    fields = SamplePlots()
    fields.initialize_plot_data(config)

    # Get parameters from params.csv
    proj_dir = os.path.dirname(ini_path)
    p_file = os.path.join(proj_dir, "params.csv")

    ps = {}
    with open(p_file) as file:
        for line in file:
            k, v = line.split(',')[0], line.split(',')[1]
            ps[k] = v
    ps.pop('Unnamed: 0', None)  # Artifact of pandas

    df = obs_field_cycle.field_day_loop(config, fields, debug_flag=debug_flag, params=ps)

    # debug_flag=False just returns the ndarray for writing
    if not debug_flag:
        eta_result, swe_result = df
        for i, fid in enumerate(fields.input['order']):
            pred_eta, pred_swe = eta_result[:, i], swe_result[:, i]
            np.savetxt(os.path.join(d, 'pest', 'pred', 'pred_eta_{}.np'.format(fid)), pred_eta)
            np.savetxt(os.path.join(d, 'pest', 'pred', 'pred_swe_{}.np'.format(fid)), pred_swe)
        end_time = time.time()
        print('\nExecution time: {:.2f} seconds'.format(end_time - start_time))

    # debug_flag=true returns a dataframe
    if debug_flag:

        targets = fields.input['order']
        first = True

        # print('\nWarning: model runner is set to debug=True, it will not write results accessible to PEST++')
        rmse_cap = []
        rmse_swe = []

        for fid in targets:

            pred_et = df[fid]['et_act'].values

            obs_et = os.path.join(proj_dir, 'obs/obs_eta_{}.np'.format(fid))
            obs_et = np.loadtxt(obs_et)
            cols = ['et_obs'] + list(df[fid].columns)
            df[fid]['et_obs'] = obs_et
            df[fid] = df[fid][cols]
            a = df[fid].loc['2010-01-01': '2021-01-01']

            comp = pd.DataFrame(data=np.vstack([obs_et, pred_et]).T, columns=['obs', 'pred'], index=df[fid].index)
            comp['eq'] = comp['obs'] == comp['pred']
            comp['capture'] = df[fid]['capture']

            rmse = np.sqrt(((pred_et - obs_et) ** 2).mean())
            end_time = time.time()

            if first:
                print('Execution time: {:.2f} seconds\n'.format(end_time - start_time))
                first = False

            # print('{}: Mean Obs: {:.2f}, Mean Pred: {:.2f}'.format(fid, obs_et.mean(), pred_et.mean()))
            # print('{}: RMS Diff: {:.4f}'.format(fid, rmse))

            comp = comp.loc[a[a['capture'] == 1.0].index]
            pred_et, obs_et = comp['pred'], comp['obs']
            rmse = np.sqrt(((pred_et - obs_et) ** 2).mean())
            print('{}: RMSE Capture Dates: {:.4f}'.format(fid, rmse))
            rmse_cap.append(rmse)  # Am I saving the correct RMSE?

            obs_swe = os.path.join(proj_dir, 'obs/obs_swe_{}.np'.format(fid))
            obs_swe = np.loadtxt(obs_swe)
            cols = ['swe_obs'] + list(df[fid].columns)
            df[fid]['swe_obs'] = obs_swe
            df[fid] = df[fid][cols]
            swe_df = df[fid].loc['2010-01-01': '2021-01-01'][['swe_obs', 'swe']]
            swe_df.dropna(axis=0, inplace=True)
            pred_swe = swe_df['swe'].values
            obs_swe = swe_df['swe_obs'].values
            rmse = np.sqrt(((pred_swe - obs_swe) ** 2).mean())
            # print('{}: RMSE SWE: {:.4f}\n'.format(fid, rmse))
            rmse_swe.append(rmse)
        return rmse_cap, rmse_swe


def run_fields_opt(pars):
    """ Variation on run_fields above, to be used with scipy.optimize.minimize.

    Main differences:
    - Pass parameters directly into the function, without changing params.csv - Can I do this?
    - Runs for a single field. - How do I change which field?
    """
    project = 'tongue'
    direc = 'C:/Users/CND571/PycharmProjects/swim-rs/examples/{}'.format(project_)
    ini_path = os.path.join(direc, '{}_swim.toml'.format(project))

    debug_flag = True

    start_time = time.time()

    config = ProjectConfig()
    config.read_config(ini_path)

    fields = SamplePlots()
    fields.initialize_plot_data(config)

    # print(fields.input['time_series'])  # way too long

    # How to pare back the number of fields run?

    # Get parameters from params.csv
    proj_dir = os.path.dirname(ini_path)
    p_file = os.path.join(proj_dir, "params.csv")

    ps = {}
    with open(p_file) as file:
        for line in file:
            k, v = line.split(',')[0], line.split(',')[1]
            ps[k] = v
    ps.pop('Unnamed: 0', None)  # Artifact of pandas
    # print(ps)

    # Update parameters for a single field
    fid = fields.input['order'][0]  # defining the single field
    par_names = ['aw', 'rew', 'tew', 'ndvi_alpha', 'ndvi_beta', 'mad', 'swe_alpha', 'swe_beta']
    # par_names = p_dict.keys()
    for i in range(8):
        k = par_names[i] + '_' + fid
        ps[k] = pars[i]

    # Run model (all fields, for now)
    df = obs_field_cycle.field_day_loop(config, fields, debug_flag=debug_flag, params=ps)

    # calculate rmse
    pred_et = df[fid]['et_act'].values

    obs_et = os.path.join(proj_dir, 'obs/obs_eta_{}.np'.format(fid))
    obs_et = np.loadtxt(obs_et)
    cols = ['et_obs'] + list(df[fid].columns)
    df[fid]['et_obs'] = obs_et
    df[fid] = df[fid][cols]
    a = df[fid].loc['2010-01-01': '2021-01-01']

    comp = pd.DataFrame(data=np.vstack([obs_et, pred_et]).T, columns=['obs', 'pred'], index=df[fid].index)
    comp['eq'] = comp['obs'] == comp['pred']
    comp['capture'] = df[fid]['capture']

    rmse = np.sqrt(((pred_et - obs_et) ** 2).mean())
    end_time = time.time()

    print('Execution time: {:.2f} seconds'.format(end_time - start_time))
    print(pars)

    # # print('{}: Mean Obs: {:.2f}, Mean Pred: {:.2f}'.format(fid, obs_et.mean(), pred_et.mean()))
    # # print('{}: RMS Diff: {:.4f}'.format(fid, rmse))

    comp = comp.loc[a[a['capture'] == 1.0].index]
    pred_et, obs_et = comp['pred'], comp['obs']
    rmse_cap = np.sqrt(((pred_et - obs_et) ** 2).mean())
    print('RMSE Capture Dates: {:.4f}'.format(rmse_cap))

    obs_swe = os.path.join(proj_dir, 'obs/obs_swe_{}.np'.format(fid))
    obs_swe = np.loadtxt(obs_swe)
    cols = ['swe_obs'] + list(df[fid].columns)
    df[fid]['swe_obs'] = obs_swe
    df[fid] = df[fid][cols]
    swe_df = df[fid].loc['2010-01-01': '2021-01-01'][['swe_obs', 'swe']]
    swe_df.dropna(axis=0, inplace=True)
    pred_swe = swe_df['swe'].values
    obs_swe = swe_df['swe_obs'].values
    rmse_swe = np.sqrt(((pred_swe - obs_swe) ** 2).mean())
    print('RMSE SWE: {:.4f}'.format(rmse_swe))

    rmse = 5*rmse_cap + rmse_swe
    print("Total RMSE: {:.4f}".format(rmse))
    return rmse


def run_fields_opt_1(pars):
    """ Variation on run_fields above, to be used with scipy.optimize.minimize.

    Main differences:
    - Pass parameters directly into the function, without changing params.csv - Can I do this?
    - Runs for a single field. - How do I change which field?
    """
    project = 'tongue'
    direc = 'C:/Users/CND571/PycharmProjects/swim-rs/examples/{}'.format(project_)
    ini_path = os.path.join(direc, '{}_swim.toml'.format(project))

    debug_flag = True

    start_time = time.time()

    # Update parameters for a single field
    par_names = ['aw', 'rew', 'tew', 'ndvi_alpha', 'ndvi_beta', 'mad', 'swe_alpha', 'swe_beta']
    # par_names = p_dict.keys()
    for i in range(8):
        k = par_names[i] + '_' + fid
        ps[k] = pars[i]

    # Run model (all fields, for now)
    df = obs_field_cycle.field_day_loop(config, fields, debug_flag=debug_flag, params=ps)

    # calculate rmse
    pred_et = df[fid]['et_act'].values

    obs_et = os.path.join(proj_dir, 'obs/obs_eta_{}.np'.format(fid))
    obs_et = np.loadtxt(obs_et)
    cols = ['et_obs'] + list(df[fid].columns)
    df[fid]['et_obs'] = obs_et
    df[fid] = df[fid][cols]
    a = df[fid].loc['2010-01-01': '2021-01-01']

    comp = pd.DataFrame(data=np.vstack([obs_et, pred_et]).T, columns=['obs', 'pred'], index=df[fid].index)
    comp['eq'] = comp['obs'] == comp['pred']
    comp['capture'] = df[fid]['capture']

    rmse = np.sqrt(((pred_et - obs_et) ** 2).mean())
    end_time = time.time()

    print('Execution time: {:.2f} seconds'.format(end_time - start_time))
    print(pars)

    # # print('{}: Mean Obs: {:.2f}, Mean Pred: {:.2f}'.format(fid, obs_et.mean(), pred_et.mean()))
    # # print('{}: RMS Diff: {:.4f}'.format(fid, rmse))

    comp = comp.loc[a[a['capture'] == 1.0].index]
    pred_et, obs_et = comp['pred'], comp['obs']
    rmse_cap = np.sqrt(((pred_et - obs_et) ** 2).mean())
    print('RMSE Capture Dates: {:.4f}'.format(rmse_cap))

    obs_swe = os.path.join(proj_dir, 'obs/obs_swe_{}.np'.format(fid))
    obs_swe = np.loadtxt(obs_swe)
    cols = ['swe_obs'] + list(df[fid].columns)
    df[fid]['swe_obs'] = obs_swe
    df[fid] = df[fid][cols]
    swe_df = df[fid].loc['2010-01-01': '2021-01-01'][['swe_obs', 'swe']]
    swe_df.dropna(axis=0, inplace=True)
    pred_swe = swe_df['swe'].values
    obs_swe = swe_df['swe_obs'].values
    rmse_swe = np.sqrt(((pred_swe - obs_swe) ** 2).mean())
    print('RMSE SWE: {:.4f}'.format(rmse_swe))

    rmse = 5*rmse_cap + rmse_swe
    print("Total RMSE: {:.4f}".format(rmse))
    return rmse


def update_params(path, nps, num_targets):
    params = pd.read_csv(os.path.join(path, 'params.csv'))
    for i in range(8):
        params.loc[i * num_targets:(i + 1) * num_targets, 'value'] = nps[i]
    params.to_csv(os.path.join(path, 'params.csv'), index=False)


def one_param_sensitivity(path, ps, which_p, targets, ini_path, param_dict, l_bound, u_bound):
    num_targets = len(targets)
    n_sens = 5
    p_idx = param_dict[which_p]
    sens = np.linspace(l_bound[p_idx], u_bound[p_idx], n_sens)
    print(which_p, sens)
    # create range of parameters
    es_cap, es_swe = np.zeros((n_sens, num_targets)), np.zeros((n_sens, num_targets))
    for i in range(n_sens):
        # iterate through range of parameters
        ps[p_idx] = sens[i]
        print(i, ps)
        update_params(path, ps, num_targets)
        # recover rmse of capture dates and swe
        e_cap, e_swe = run_fields(ini_path=ini_path, debug_flag=True)
        # print(i, )
        es_cap[i] = e_cap
        es_swe[i] = e_swe

    plt.figure()
    for i in range(num_targets):
        if which_p in ['swe_alpha', 'swe_beta']:
            plt.subplot(2, int(num_targets / 2), i + 1)
            plt.title(targets[i])
            if i >= int(num_targets / 2):
                plt.xlabel(which_p)
            if i % int(num_targets / 2) == 0:
                plt.ylabel('RMSE for SWE')
            plt.plot(sens, es_swe[:, i])
            plt.ylim(np.min(es_swe), np.max(es_swe))
            plt.grid()
        else:
            plt.subplot(2, int(num_targets / 2), i + 1)
            plt.title(targets[i])
            if i >= int(num_targets / 2):
                plt.xlabel(which_p)
            if i % int(num_targets / 2) == 0:
                plt.ylabel('RMSE for ET on Capture Dates')
            plt.plot(sens, es_cap[:, i])
            plt.ylim(np.min(es_cap), np.max(es_cap))
            plt.grid()
    # plt.tight_layout()
    plt.show()


def sensitivity(path, ps, targets, ini_path, l_bound, u_bound, save=False, plot_tar=None, n_sens=3):
    """ Do a sensitivity analysis from a base set of parameters.
    params:
    path: str; directory where control files live. Needs to have 'params.csv'.
    ps: list; base parameter set from which to base the sensitivity analysis. all targets will have same parameters.
    targets: list; list of field ids
    ini_path: str; filepath pointing to the .toml control file needed to run the swim model.
    field_type: str, optional; parameter to run swim.
    project: str, optional; parameter to run swim, project identifier.
    save: bool, optional; whether to save results to a txt file in 'path'.
    plot_tar: int, optional, default None; if int given, plot results for target[plot_tar-1], otherwise,
    no plots will be generated. Note required increase in index, as zero will plot nothing instead of the first target.
    n_sens: int, optional; number of values to include in sensitivity analysis. model will be run 8*n_sens times.
    """
    num_targets = len(targets)
    ip_dict = {0: 'aw', 1: 'rew', 2: 'tew', 3: 'ndvi_alpha', 4: 'ndvi_beta', 5: 'mad', 6: 'swe_alpha', 7: 'swe_beta'}

    rmses = []

    update_params(path, ps, num_targets)
    ce_cap, ce_swe = run_fields(ini_path=ini_path, debug_flag=True)
    rmses.append(ce_cap)
    rmses.append(ce_swe)

    # create range of parameters
    sens = np.zeros((8, n_sens))
    es_cap, es_swe = np.zeros((8, n_sens, num_targets)), np.zeros((8, n_sens, num_targets))
    for i in range(8):
        sens[i] = np.linspace(l_bound[i], u_bound[i], n_sens)
        print(ip_dict[i], sens[i], ps)
        temp = ps.copy()
        for j in range(n_sens):
            # iterate through range of parameters
            temp[i] = sens[i][j]
            # print(ip_dict[i], j, temp)
            update_params(path, temp, num_targets)
            # recover rmse of capture dates and swe
            e_cap, e_swe = run_fields(ini_path=ini_path, debug_flag=True)
            es_cap[i, j] = e_cap
            es_swe[i, j] = e_swe
            rmses.append(e_cap)
            rmses.append(e_swe)

    if save:
        np.savetxt(os.path.join(path, 'sensitivity_results.txt'), np.asarray(rmses))
        np.savetxt(os.path.join(path, 'sensitivity_base_params.txt'), ps)

    if plot_tar:
        which_tar = plot_tar - 1
        plt.figure()
        plt.suptitle("Field: {}".format(targets[which_tar]))
        for i in range(8):
            p = ip_dict[i]
            plt.subplot(2, 4, i + 1)
            plt.xlabel(p)
            if i < 6:  # swe
                plt.ylabel('RMSE for ET on Capture Dates')
                plt.vlines(ps[i], np.min(es_cap), np.max(es_cap), label='Base Value: {:.2f}'.format(ps[i]),
                           colors='tab:grey', linestyles='dashed')
                plt.hlines(ce_cap[which_tar], lb[i], ub[i], label='RMSE: {:.2f}'.format(ce_cap[which_tar]),
                           colors='tab:orange', linestyles='dashed')
                plt.plot(sens[i], es_cap[i, :, which_tar])
                plt.scatter(ps[i], ce_cap[which_tar], zorder=3)
                plt.xlim(lb[i], ub[i])
                plt.ylim(np.min(es_cap), np.max(es_cap))
            else:  # et
                plt.ylabel('RMSE for SWE')
                plt.vlines(ps[i], np.min(es_swe), np.max(es_swe), label='Base Value: {:.2f}'.format(ps[i]),
                           colors='tab:grey', linestyles='dashed')
                plt.hlines(ce_swe[which_tar], lb[i], ub[i], label='RMSE: {:.2f}'.format(ce_swe[which_tar]),
                           colors='tab:orange', linestyles='dashed')
                plt.plot(sens[i], es_swe[i, :, which_tar])
                plt.scatter(ps[i], ce_swe[which_tar], zorder=3)
                plt.xlim(lb[i], ub[i])
                plt.ylim(np.min(es_swe), np.max(es_swe))
                plt.yscale('log')
            plt.grid(zorder=-2)
            plt.legend()


def plot_things(file_dir, targets, which_tar, l_bound, u_bound):
    ip_dict = {0: 'aw', 1: 'rew', 2: 'tew', 3: 'ndvi_alpha', 4: 'ndvi_beta', 5: 'mad', 6: 'swe_alpha', 7: 'swe_beta'}

    ps = np.genfromtxt(os.path.join(file_dir, 'sensitivity_base_params.txt'))

    rmses = np.genfromtxt(os.path.join(file_dir, 'sensitivity_results.txt'))
    n_sens = int((len(rmses)-2)/16)
    ce_cap = rmses[0]
    ce_swe = rmses[1]
    es_cap = rmses[2::2]
    es_swe = rmses[3::2]

    # print(np.shape(es_cap))
    # print(np.shape(es_swe))

    # create range of parameters
    sens = np.zeros((8, n_sens))
    for i in range(8):
        sens[i] = np.linspace(l_bound[i], u_bound[i], n_sens)

    # print(es_cap)
    # print(es_swe)

    plt.figure()
    plt.suptitle("Field: {}".format(targets[which_tar]))
    for i in range(8):
        p = ip_dict[i]
        plt.subplot(2, 4, i + 1)
        plt.xlabel(p)
        if i < 6:  # swe
            plt.ylabel('RMSE for ET on Capture Dates')
            # plt.scatter(ps[i], ce_cap[which_tar], zorder=3,
            #             label='Base Value: {:.2f}, RMSE: {:.2f}'.format(ps[i], ce_cap[which_tar]))
            plt.vlines(ps[i], np.min(es_cap), np.max(es_cap), label='Base Value: {:.2f}'.format(ps[i]),
                       colors='tab:grey', linestyles='dashed')
            plt.hlines(ce_cap[which_tar], lb[i], ub[i], label='RMSE: {:.2f}'.format(ce_cap[which_tar]),
                       colors='tab:orange', linestyles='dashed')
            plt.plot(sens[i], es_cap[i * n_sens:i * n_sens + n_sens, which_tar])
            plt.scatter(ps[i], ce_cap[which_tar], zorder=3)  # , label='Base Value: {:.2f}'.format(ps[i]))
            plt.xlim(lb[i], ub[i])
            plt.ylim(np.min(es_cap), np.max(es_cap))
        else:  # et
            plt.ylabel('RMSE for SWE')
            # plt.scatter(ps[i], ce_swe[which_tar], zorder=3,
            #             label='Base Value: {:.2f}, RMSE: {:.2f}'.format(ps[i], ce_swe[which_tar]))
            plt.vlines(ps[i], np.min(es_swe), np.max(es_swe), label='Base Value: {:.2f}'.format(ps[i]),
                       colors='tab:grey', linestyles='dashed')
            plt.hlines(ce_swe[which_tar], lb[i], ub[i], label='RMSE: {:.2f}'.format(ce_swe[which_tar]),
                       colors='tab:orange', linestyles='dashed')
            plt.plot(sens[i], es_swe[i * n_sens:i * n_sens + n_sens, which_tar])
            plt.scatter(ps[i], ce_swe[which_tar], zorder=3)  # , label='Base Value: {:.2f}'.format(ps[i]))
            plt.xlim(lb[i], ub[i])
            plt.ylim(np.min(es_swe), np.max(es_swe))
            plt.yscale('log')
        # plt.yscale('log')
        plt.grid(zorder=-2)
        plt.legend()
    # plt.tight_layout()  # Why is this so bad?
    # plt.show()


if __name__ == '__main__':
    # project_ = 'tongue'
    # d = 'C:/Users/CND571/PycharmProjects/swim-rs/examples/{}'.format(project_)
    project_ = info.project_name
    d = info.d
    ini = os.path.join(d, '{}_swim.toml'.format(project_))

    # Warning: order in file is not printed order.
    # Order in file is: aw, rew, tew, ndvi_alpha, ndvi_beta, mad, swe_alpha, swe_beta
    p_dict = {'aw': 0, 'rew': 1, 'tew': 2, 'ndvi_alpha': 3, 'ndvi_beta': 4, 'mad': 5, 'swe_alpha': 6, 'swe_beta': 7}
    lb = [15.0, 2.0, 6.0, -0.7, 0.5, 0.1, -0.7, 0.5]
    ub = [700.0, 6.0, 29.0, 1.5, 1.7, 0.9, 1.5, 1.7]
    tars = [1779, 1787, 1793, 1797, 1801, 1804]

    # Declare parameters (change params.csv)
    # new_params = [145.0, 3.0, 18.0, 0.2, 1.25, 0.6, 0.07, 1.0]  # original values
    # new_params = np.asarray([145.0, 3.0, 18.0, 0.0, 1.25, 0.6, 0.5, 1.0])

    # # Testing a single run
    # update_params(d, new_params, len(tars))
    run_fields(ini_path=ini, debug_flag=True)

    # # random starting point.
    # random.seed(23)
    # first_params = np.asarray([lb[i] + random.random() * (ub[i] - lb[i]) for i in range(8)])
    #
    # # setup
    # config = ProjectConfig()
    # config.read_config(ini)
    # fields = SamplePlots()
    # fields.initialize_plot_data(config)
    # # declaring single field
    # fid = fields.input['order'][0]
    # # Can I change only fields? - Yes, that appears to work! I did also change a line in the obs_field_cycle code.
    # # "cropping" input dictionaries
    # fields.input['props'] = {fid: fields.input['props'][fid]}
    # fields.input['irr_data'] = {fid: fields.input['irr_data'][fid]}
    # fields.input['order'] = [fid]
    # # the last one would need to cut short the list of variables at the deepest level of the dictionary...
    # # fields.input['time_series'] =
    # for k1, v1 in fields.input['time_series'].items():
    #     for k2, v2 in v1.items():
    #         if k2 != 'doy':
    #             fields.input['time_series'][k1][k2] = [v2[0]]
    # # Get parameters from params.csv
    # proj_dir = os.path.dirname(ini)
    # p_file = os.path.join(proj_dir, "params.csv")
    # ps = {}
    # with open(p_file) as file:
    #     for line in file:
    #         k, v = line.split(',')[0], line.split(',')[1]
    #         ps[k] = v
    # ps.pop('Unnamed: 0', None)  # Artifact of pandas
    # # print(ps)
    #
    # print("all fields:")
    # run_fields_opt(new_params)
    # print()
    #
    # print("one field:")
    # run_fields_opt_1(new_params)
    # print()

    # Sensitivity Analyses
    # one_param_sensitivity(d, new_params, 'ndvi_alpha', tars, ini, p_dict, lb, ub)
    # sensitivity(d, new_params, tars, ini, lb, ub, save=True, n_sens=5)
    # plot_things(d, tars, 0, lb, ub)

    # # Plot all targets/fields
    # for i in range(len(tars)):
    #     plot_things(d, tars, i, lb, ub)

    # trying to use optimization
    # I need to be doing this individually for each field? Yup.
    # SO take the function and return a single value (combine cap and swe rmse's, how?)
    # what function should I be using? - it should take the initial guess, and return the rmse value to optimize.

    # big_start_time = time.time()
    # # what is the default tolerance?
    # res = minimize(run_fields_opt_1, x0=first_params, method='Nelder-Mead', bounds=Bounds(lb=lb, ub=ub), tol=2,
    #                options={'disp': True, 'maxiter': 200})
    # # 'maxiter': 100  # This works, it just seems like a very crude way of stopping it.
    # # 'fatol': 0.01  # This does not seem to be doing anything. even with a value of 1, it runs forever.
    # #  'return_all': True  # This causes function to return an array of all of the simplexes at the end.
    # # What tolerance is being set? 1 seems too high, but it is stabilizing at a function value aroung 10.11,
    # # out to 2 decimal places.
    # big_end_time = time.time()
    # print()
    # print("Time to run: {:.2f}".format(big_end_time - big_start_time))
    # print("Results: ")
    # print(res)

    # outer loop with fields and config files?
    # then calibrate model for each field using minimize.

    plt.show()
