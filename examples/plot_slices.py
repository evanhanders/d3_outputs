"""
This script plots snapshots of the evolution of 2D slices from a 2D simulation in polar geometry.

The fields specified in 'fig_type' are plotted (temperature and enstrophy by default).
To plot a different set of fields, add a new fig type number, and expand the fig_type if-statement.

Usage:
    plot_annular_snapshots.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: slices]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Total number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 3]
    --row_inch=<in>                     Number of inches / row [default: 3]
    --static_cbar                       If flagged, don't evolve the colorbar with time

    --fig_type=<fig_type>               Type of figure to plot
                                            1 - T, u
                                            2 - u, ω, ωfluc
                                            3 - partial pressures
                                        [default: 1]
"""
from docopt import docopt
args = docopt(__doc__)
from plotpal.slices import SlicePlotter

# Read in master output directory
root_dir    = args['<root_dir>']
data_dir    = args['--data_dir']
if root_dir is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

# Read in additional plot arguments
start_fig   = int(args['--start_fig'])
start_file  = int(args['--start_file'])
n_files     = args['--n_files']
if n_files is not None: 
    n_files = int(n_files)

# Equatorial plots
fig_name    = 'snapshots_eq'
plotter = SlicePlotter(root_dir, file_dir=data_dir, fig_name=fig_name, start_file=start_file, n_files=n_files)
plotter_kwargs = { 'col_in' : int(args['--col_inch']), 'row_in' : int(args['--row_inch']) }
if int(args['--fig_type']) == 1:
    plotter.setup_grid(2, 2, polar=True, **plotter_kwargs)
    fnames = [  (('T_eq',),         {'polar' : True, 'remove_x_mean' : True}), 
                (('u_eq',),        {'vector_ind' : 0, 'polar' : True, 'cmap' : 'PuOr_r'}),
                (('z_vort_eq',),        {'polar' : True, 'cmap' : 'PiYG_r'}),
                (('u_eq',),        {'vector_ind' : 2, 'polar' : True, 'cmap' : 'PuOr_r'}),
             ]
for tup in fnames:
    plotter.add_colormesh(*tup[0], x_basis='phi_0', y_basis='r_0', **tup[1])
plotter.plot_colormeshes(start_fig=start_fig, dpi=int(args['--dpi']))

# Orthographic plots
fig_name    = 'snapshots_ortho'
plotter = SlicePlotter(root_dir, file_dir=data_dir, fig_name=fig_name, start_file=start_file, n_files=n_files)
plotter_kwargs = { 'col_in' : int(args['--col_inch']), 'row_in' : int(args['--row_inch']) }
if int(args['--fig_type']) == 1:
    plotter.setup_grid(1, 3, ortho=True, **plotter_kwargs)
    fnames = [  (('T_r0.95',),         {'ortho' : True, 'remove_x_mean' : True}), 
                (('u_r0.95',),         {'vector_ind': 2, 'ortho' : True, 'cmap' : 'PuOr_r'}),
                (('z_vort_r0.95',),         {'ortho' : True, 'cmap' : 'PiYG_r'}),
             ]
for tup in fnames:
    plotter.add_colormesh(*tup[0], x_basis='phi_0', y_basis='theta_0', **tup[1])
plotter.plot_colormeshes(start_fig=start_fig, dpi=int(args['--dpi']))

# Create Plotter object, tell it which fields to plot
fig_name    = 'snapshots_mer'
plotter = SlicePlotter(root_dir, file_dir=data_dir, fig_name=fig_name, start_file=start_file, n_files=n_files)
plotter_kwargs = { 'col_in' : int(args['--col_inch']), 'row_in' : int(args['--row_inch']) }
if int(args['--fig_type']) == 1:
    plotter.setup_grid(2, 3, polar=True, **plotter_kwargs)
    fnames = [  (('z_vort(phi=0)',),         {'meridional' : True, 'cmap' : 'PuOr_r'}), 
                (('z_vort(phi=pi)',),        {'meridional' : True, 'cmap' : 'PuOr_r'}),
                (('T(phi=0)',),         {'meridional' : True, 'cmap' : 'RdBu_r', 'remove_x_mean': True}), 
                (('T(phi=pi)',),        {'meridional' : True, 'cmap' : 'RdBu_r'}),
                (('u(phi=0)',),          {'meridional' : True, 'vector_ind' : 0, 'cmap' : 'PiYG_r'}),
                (('u(phi=pi)',),         {'meridional' : True, 'vector_ind' : 0, 'cmap' : 'PiYG_r'}),
             ]
for tup in fnames:
    plotter.add_colormesh(*tup[0], x_basis='theta_0', y_basis='r_0', **tup[1])
plotter.plot_colormeshes(start_fig=start_fig, dpi=int(args['--dpi']))
