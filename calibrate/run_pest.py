import os
import shutil

from pyemu import os_utils


def run_pst(_dir, _cmd, pst_file, num_workers, worker_root, master_dir=None, verbose=True, cleanup=True):

    try:
        os.chdir(worker_root)
        [print('rmtree: {}'.format(os.path.join(worker_root, d))) for d in os.listdir(worker_root)]
        [shutil.rmtree(os.path.join(worker_root, d)) for d in os.listdir(worker_root)]
    except FileNotFoundError:
        os.mkdir(os.path.join(worker_root))
        if master_dir:
            os.mkdir(os.path.join(master_dir))

    os_utils.start_workers(_dir, _cmd, pst_file,
                           num_workers=num_workers,
                           worker_root=worker_root,
                           verbose=verbose,
                           master_dir=master_dir,
                           cleanup=cleanup,
                           port=5000)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    root = os.path.join(home, 'PycharmProjects', 'swim-rs')
    project_ws = os.path.join(root, 'tutorials', '2_Fort_Peck')

    p_dir = os.path.join(project_ws, 'pest')
    m_dir = os.path.join(project_ws, 'master')
    w_dir = os.path.join(project_ws, 'workers')
    exe_ = 'pestpp-ies'

    _pst = os.path.join(project_ws, 'pest', '2_Fort_Peck.pst')

    _workers = 4

    run_pst(p_dir, exe_, _pst, num_workers=_workers, worker_root=w_dir,
            master_dir=m_dir, verbose=False)
# ========================= EOF ====================================================================
