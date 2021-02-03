#!/usr/bin/env python3

"""
MODULE:       r.viewshed.impact

AUTHOR(S):    Zofie Cimburova, Stefan Blumentrath

PURPOSE:      TODO1 add purpose description

COPYRIGHT:    (C) 2021 by Zofie Cimburova, Stefan Blumentrath, and the GRASS Development Team

REFERENCES:   TODO1 reference papers used and the paper to be published

# TODO1 which licence text to use?
This program is free software under the GNU General Public
License (>=v2). Read the file COPYING that comes with GRASS
for details.
"""

#%module
#% label: Visual impact of defined exposure source
#% description: Computes visual impact of defined exposure source using weighted parametrised viewshed analysis
#% keyword: raster
#% keyword: viewshed
#% keyword: line of sight
#% keyword: LOS
#% keyword: exposure
#% keyword: impact
#%end

#%option G_OPT_V_INPUT
#% key: exposure_source
#% label: Name of input map of exposure source locations
#% description: Name of input map of exposure source locations
#%end

#%option
#% key: attribute
#% description: Name of attribute column to store visual impact values
#%end

#%option G_OPT_R_INPUT
#% key: weight
#% description: Name of input weights raster map
#%end

#%option G_OPT_R_INPUT
#% key: dsm
#% description: Name of input digital surface raster map
#%end

#%flag
#% key: g
#% description: Exposure source dimensions influence visibility impact
#%end

#%flag
#% key: c
#% description: Consider the curvature of the earth (current ellipsoid)
#% guisection: Viewshed settings
#%end

#%option
#% key: observer_elevation
#% type: double
#% required: no
#% key_desc: value
#% description: Observer elevation above the ground (value >= 0.0)
#% answer: 1.5
#% guisection: Viewshed settings
#%end

#%option
#% key: max_distance
#% type: double
#% required: no
#% key_desc: value
#% description: Maximum visibility radius (value > 0.0). -1 for infinity
#% answer: 100
#% guisection: Viewshed settings
#%end

#%option
#% key: approach
#% type: string
#% required: no
#% options: binary, distance_decay, fuzzy_viewshed, vertical_angle, solid_angle
#% key_desc: name
#% description: Approach for viewshed parametrisation
#% guisection: Viewshed settings
#% answer: binary
#%end

#%option
#% key: b1_distance
#% type: double
#% required: no
#% key_desc: value
#% description: Radius around the viewpoint where clarity is perfect. Used in fuzzy viewshed approach.
#% guisection: Viewshed settings
#% answer: 10
#%end

#%option
#% key: sample_density
#% type: double
#% required: no
#% options: 0.0-100.0
#% key_desc: value
#% description: Density of sampling points
#% guisection: Sampling settings
#% answer: 30
#%end

#%option
#% key: seed
#% type: integer
#% required: no
#% options: 0-
#% key_desc: value
#% description: Random seed, default [random]
#% guisection: Sampling settings
#%end

#%flag
#% key: r
#% description: Consider the effect of atmospheric refraction
#% guisection: Refraction
#%end

#%option
#% key: refraction_coeff
#% type: double
#% required: no
#% key_desc: value
#% options: 0.0-1.0
#% description: Refraction coefficient
#% answer: 0.14286
#% guisection: Refraction
#%end

#%option
#% key: memory
#% type: integer
#% required: no
#% key_desc: value
#% options: 1-
#% description: Amount of memory to use in MB
#% answer: 500
#%end

#%option
#% key: cores
#% type: integer
#% required: no
#% key_desc: value
#% options: 1-
#% description: Number of cores to use in parrallelization
#% answer: 1
#%end

import os
import math
import time
import atexit
import sys
import subprocess
import numpy as np

from grass.pygrass.raster import raster2numpy
from grass.pygrass.raster import numpy2raster

from grass.pygrass.gis.region import Region
from grass.pygrass.vector.basic import Bbox
from grass.pygrass.vector import VectorTopo
from grass.pygrass.modules import Module


import grass.script as grass
from grass.script import utils as grassutils

# enable coordinate systems with various axis orientation (now assuming Y-north, X-east)
#      > TODO2 is it so that X is always to the east and Y to the north? I've checked with e.g. s-jtsk and it seems so
#       > Stefan asks
# cleaning .tmp mapset
#      > TODO2 necessary to remove temporary regions? (Are not saved.)
#      > clean_temp() cleans /.tmp directory.
#           > TODO2 when to use it? In loop, or at the end?
#           > copied and adjusted function clean_temp() from https://grass.osgeo.org/grass78/manuals/libpython/_modules/script/setup.html to supress warnings
#           > Stefan checks - might be issue for parallelization

# global variables
TMP_RAST = []  # to collect temporary rasters
TMP_VECT = []  # to collect temporary vectors


def cleanup():
    """Remove raster and vector maps stored in a list"""
    for rast in TMP_RAST:
        grass.run_command(
            'g.remove',
            flags='f',
            type='raster',
            name=rast,
            quiet=True,
            stderr=subprocess.PIPE
        )
    for vect in TMP_VECT:
        grass.run_command(
            'g.remove',
            flags='f',
            type='vector',
            name=vect,
            quiet=True,
            stderr=subprocess.PIPE
        )


def call(cmd, **kwargs):
    """Wrapper for subprocess.call to deal with platform-specific issues
    Function copied from
    https://grass.osgeo.org/grass78/manuals/libpython/_modules/script/setup.html
    to enable supressing message from clean_temp()"""

    windows = sys.platform == 'win32'

    if windows:
        kwargs['shell'] = True
    return subprocess.call(cmd, **kwargs)


def clean_temp():
    """Modified from
    https://grass.osgeo.org/grass78/manuals/libpython/_modules/script/setup.html
    to enable supressing message"""

    # gcore.message(_('Cleaning up temporary files...'))
    nul = open(os.devnull, 'w')
    gisbase = os.environ['GISBASE']
    call([os.path.join(gisbase, 'etc', 'clean_temp')], stdout=nul)
    nul.close()


def distance_decay_reverse(lreg_shape, t_loc, np_viewshed):
    """Calculates distance decay weights to target based on
    Gret-Regamey et al. (2007) and Chamberlain & Meitner (2013) and use these
    to parametrise binary viewshed
    :param lreg_shape: Dimensions of local computational region
    :type lreg_shape: list
    :param t_loc: Array of target point coordinates in local coordinate system
    :type t_loc: ndarray
    :param np_viewshed: 2D array of binary viewshed
    :type np_viewshed: ndarray
    :param NSRES: Cell resolution in N-S direction
    :type NSRES: float
    :param EWRES: Cell resolution in E-W direction
    :type EWRES: float
    :return: 2D array of weighted parametrised viewshed
    :rtype: ndarray
    """
    # 1. local row, col coordinates of observer points V
    #    2D array (lreg_shape[0] x lreg_shape[1] x 2)
    v_loc = np.array(
        [
            np.tile(
                np.arange(0.5, lreg_shape[0] + 0.5).reshape(-1, 1), (1, lreg_shape[1])
            ),
            np.tile(
                np.arange(0.5, lreg_shape[1] + 0.5).reshape(-1, 1).transpose(), (lreg_shape[0], 1)
            ),
        ]
    )

    # 2. vector VT, adjusted for cell size
    #    2D array (lreg_shape[0] x lreg_shape[1])
    v_vect = np.array(
        [
            (v_loc[0] - t_loc[0]) * NSRES,
            (v_loc[1] - t_loc[1]) * EWRES
        ]
    )

    # 3. size of vector VT
    #    2D array (lreg_shape[0] x lreg_shape[1])
    v_scal = np.sqrt(v_vect[0] ** 2 + v_vect[1] ** 2)

    # replace 0 distance for central pixel by 1 to avoid division by 0
    v_scal = np.where(v_scal == 0, 1, v_scal)

    # 4. distance decay function
    distance_decay = 1 / (v_scal ** 2)

    # 5. Multiply distance decay by binary viewshed and weight
    weight = t_loc[-1]
    np_viewshed_param = distance_decay * np_viewshed * weight

    return np_viewshed_param


def fuzzy_viewshed_reverse(lreg_shape, t_loc, np_viewshed):
    """Calculates fuzzy viewshed weights from viewpoints to target based on
    Fisher (1994) and use these to parametrise binary viewshed
    :param lreg_shape: Dimensions of local computational region
    :type lreg_shape: list
    :param t_loc: Array of target point coordinates in local coordinate system
    :type t_loc: ndarray
    :param np_viewshed: 2D array of binary viewshed
    :type np_viewshed: ndarray
    :param B_1: Radius of zone around the viewpoint where clarity is perfect
    :type B_1: float
    :param MAX_DIST: Radius of crossover point (maximum viewshed distance)
    :type MAX_DIST: float
    :param NSRES: Cell resolution in N-S direction
    :type NSRES: float
    :param EWRES: Cell resolution in E-W direction
    :type EWRES: float
    :return: 2D array of weighted parametrised viewshed
    :rtype: ndarray
    """
    # 1. local row, col coordinates of observer points V
    #    2D array (lreg_shape[0] x lreg_shape[1] x 2)
    v_loc = np.array(
        [
            np.tile(
                np.arange(0.5, lreg_shape[0] + 0.5).reshape(-1, 1), (1, lreg_shape[1])
            ),
            np.tile(
                np.arange(0.5, lreg_shape[1] + 0.5).reshape(-1, 1).transpose(), (lreg_shape[0], 1)
            ),
        ]
    )

    # 2. vector VT, adjusted for cell size
    #    2D array (lreg_shape[0] x lreg_shape[1])
    v_vect = np.array(
        [
            (v_loc[0] - t_loc[0]) * NSRES,
            (v_loc[1] - t_loc[1]) * EWRES
        ]
    )

    # 3. size of vector VT
    #    2D array (lreg_shape[0] x lreg_shape[1])
    v_scal = np.sqrt(v_vect[0] ** 2 + v_vect[1] ** 2)

    # replace 0 distance for central pixel by 1 to avoid division by 0
    v_scal = np.where(v_scal == 0, 1, v_scal)

    # 4. fuzzy viewshed function
    fuzzy_viewshed = np.where(
        v_scal <= B_1, 1, 1 / (1 + ((v_scal - B_1) / MAX_DIST) ** 2)
        )

    # 5. Multiply fuzzy viewshed by binary viewshed and weight
    weight = t_loc[-1]
    np_viewshed_param = fuzzy_viewshed * np_viewshed * weight

    return np_viewshed_param


def sample_raster_with_points(r_map, cat, nsample, min_d, v_sample, seed):
    """Random sample exposure source by vector points
    :param r_map: Raster map to be sampled from
    :type r_map: string
    :param cat: Category of raster map to be sampled from
    :type cat: string
    :param nsample: Number of points to sample
    :type nsample: int
    :param min_d: Minimum distance between sampling points
    :type min_d: float
    :param v_sample: Name of output vector map of sampling points
    :type v_sample: string
    :param seed: Random seed
    :param seed: int
    :return: Name of output vector map of sampling points
    :rtype: string
    """
    # mask categories of raster map to be sampled from
    grass.run_command(
        'r.mask',
        raster=r_map,
        maskcats=cat,
        overwrite=True, # TODO2 what to do if mask already exists?
        quiet=True
    )

    # random sample points - raster
    r_sample = 'temp_rand_pts_rast'
    grass.run_command(
        'r.random.cells',
        output=r_sample,
        ncells=nsample,
        distance=min_d,
        overwrite=True,
        quiet=True,
        seed=seed
    )
    TMP_RAST.append(r_sample)

    # vectorize raster of random sample points
    grass.run_command(
        'r.to.vect',
        flags='b',
        input=r_sample,
        output=v_sample,
        type='point',
        overwrite=True,
        quiet=True,
        stderr=subprocess.PIPE
    )

    # remove mask
    grass.run_command(
        'r.mask',
        flags='r',
        quiet=True
    )

    # remove random sample points - raster
    grass.run_command(
        'g.remove',
        flags='f',
        type='raster',
        name=r_sample,
        quiet=True
    )

    return v_sample


def txt2numpy(
    tablestring,
    sep=',',
    names=None,
    null_value=None,
    fill_value=None,
    comments='#',
    usecols=None,
    encoding=None,
    structured=True,
    ):
    """
    Taken from #TODO link
    Can be removed when the function is included in grass core.
    Read table-like output from grass modules as Numpy array;
    format instructions are handed down to Numpys genfromtxt function
    :param stdout: tabular stdout from GRASS GIS module call
    :type stdout: str|byte
    :param sep: Separator delimiting columns
    :type sep: str
    :param names: List of strings with names for columns
    :type names: list
    :param null_value: Characters representing the no-data value
    :type null_value: str
    :param fill_value: Value to fill no-data with
    :type fill_value: str
    :param comments: Character that identifies comments in the input string
    :type comments: str
    :param usecols: List of columns to import
    :type usecols: list
    :param structured: return structured array if True, un-structured otherwise
    :type structured: bool
    :return: numpy.ndarray
        >>> import grass.script.core as grass
        >>> import numpy as np
        >>> txt = grass.read_command("r.stats", flags="cn", input="basin_50K,geology_30m", separator="|")
        >>> np_array = txt2numpy(txt, sep="|", names=None)
        >>> print(np_array)
    """

    from io import BytesIO

    if not encoding:
        encoding = grassutils._get_encoding()

    if type(tablestring).__name__ == 'str':
        tablestring = grass.encode(tablestring, encoding=encoding)
    elif type(tablestring).__name__ != 'bytes':
        raise GrassError(_('Unsupported data type'))

    kwargs = {
        'missing_values': null_value,
        'filling_values': fill_value,
        'usecols': usecols,
        'names': names,
        #'encoding': encoding, # TODO1 returns TypeError: genfromtxt() got an unexpected keyword argument 'encoding'
        'delimiter': sep
        }

    if structured:
        kwargs['dtype'] = None

    np_array = np.genfromtxt(
        BytesIO(tablestring),
        **kwargs
    )
    return np_array


def main():

    # set numpy printing options
    np.set_printoptions(formatter={'float': lambda x: '{0:0.2f}'.format(x)})

    # ==========================================================================
    # Input data
    # ==========================================================================
    ## DSM
    global R_DSM
    R_DSM = options['dsm']

    # test if exist
    gfile_dsm = grass.find_file(name=R_DSM, element='cell')
    if not gfile_dsm['file']:
        grass.fatal('Raster map <%s> not found' % R_DSM)

    ## Exposure source
    sources = options['exposure_source'].split("@")[0] #TODO how to check that it's a polygon map?

    # test if exist
    if sources:
        gfile_source = grass.find_file(name=sources, element='vector')
        if not gfile_source['file']:
            grass.fatal('Vector map <%s> not found' % sources)

    v_sources = VectorTopo(sources)
    v_sources.open('rw')
    no_sources = v_sources.num_primitive_of('area')

    ## Weights
    r_weights = options['weight']

    use_weights = 0
    gfile_weights = grass.find_file(name=r_weights, element='cell')
    if gfile_weights['file']:
        use_weights = 1

    ## Attribute to store visual impact values
    attr_vi = options['attribute']

    if attr_vi in v_sources[1].attrs.keys(): # TODO how to check if attribute already exists?
        grass.fatal('Attribute <%s> already exists' % attr_vi)
    else:
        grass.run_command(
            'v.db.addcolumn',
            map=sources,
            columns='{} {}'.format(attr_vi, 'double'))

    ## Viewshed settings
    global FLAGSTRING
    FLAGSTRING = ''
    if flags['r']:
        FLAGSTRING += 'r'
    if flags['c']:
        FLAGSTRING += 'c'

    global V_ELEVATION
    global B_1
    global REFR_COEFF
    V_ELEVATION = float(options['observer_elevation'])
    max_dist_inp = float(options['max_distance'])
    approach = options['approach']
    B_1 = float(options['b1_distance'])
    REFR_COEFF = float(options['refraction_coeff'])

    # test values
    if V_ELEVATION < 0.0:
        grass.fatal('Observer elevation must be larger than or equal to 0.0.')
    if max_dist_inp <= 0.0 and max_dist_inp != -1:
        grass.fatal('Maximum visibility radius must be larger than 0.0.')
    if approach == 'fuzzy_viewshed' and max_dist_inp == -1:
        grass.fatal('Maximum visibility radius cannot be infinity for fuzzy viewshed approch.')
    if approach == 'fuzzy_viewshed' and B_1 > max_dist_inp:
        grass.fatal(
            'Maximum visibility radius must be larger than radius around the viewpoint where clarity is perfect.'
        )

    ## Sampling settings
    source_sample_density = float(options['sample_density'])
    seed = options['seed']

    if not seed: # if seed is not set, set it to process number
        seed = os.getpid()

    ## Optional
    global MEMORY
    MEMORY = int(options['memory'])
    cores = int(options['cores'])

    # ==========================================================================
    # Region settings
    # ==========================================================================
    # check that location is not in lat/long
    if grass.locn_is_latlong():
        grass.fatal('The analysis is not available for lat/long coordinates.')

    # store the current region settings
    grass.use_temp_region()

    # get comp. region parameters
    global REG
    REG = Region()
    gl_reg_rows, gl_reg_cols = REG.rows, REG.cols
    global GL_REG_N
    global GL_REG_S
    GL_REG_N, GL_REG_S = REG.north, REG.south
    global GL_REG_E
    global GL_REG_W
    GL_REG_E, GL_REG_W = REG.east, REG.west
    global NSRES
    global EWRES
    NSRES, EWRES = REG.nsres, REG.ewres

    # check that NSRES equals EWRES
    if NSRES != EWRES:
        grass.fatal('Variable north-south and east-west 2D grid resolution is not supported')

    # adjust maximum distance as a multiplicate of region resolution
    # if infinite, set maximum distance to the max of region size
    global MAX_DIST
    if max_dist_inp != -1:
        multiplicate = math.floor(max_dist_inp / NSRES)
        MAX_DIST = multiplicate * NSRES
    else:
        max_dist_inf = max(REG.north - REG.south, REG.east - REG.west)
        multiplicate = math.floor(max_dist_inf / NSRES)
        MAX_DIST = multiplicate * NSRES

    # ==========================================================================
    # Random sample exposure source with source points T
    # ==========================================================================
    # TODO - either here, or in loop (issue 1)

    # ==========================================================================
    # Iterate over exposure source polygons
    # and calculate their visual impact
    # using weighted cumulative parametrised viewshed
    # ==========================================================================
    # counter to print progress in percentage
    counter = 0
    start_2 = time.time()
    grass.verbose(_('Iterating over trees...'))

    for v_source in v_sources.viter('areas'):
        ## Display a progress info message
        grass.percent(counter, no_sources, 1)

        ## Only process features which have attribute table
        #TODO what are the features without attributes?
        if v_source.attrs is None:
            grass.verbose("Problem")

        else:
            source_id = v_source.attrs['CrownID']
            grass.verbose('Processing exposure source ID: {}'.format(source_id))

            # ==========================================================================
            # Adjust computational region to max_dist_inp around processed exposure source
            # ==========================================================================
            # TODO

            # ==========================================================================
            # Random sample exposure source with source points T
            # ==========================================================================
            # TODO Issue 1 - either here, or before loop
            v_sampling_points = #TODO make this a temporary file

            # ==========================================================================
            # Calculate cummulative (parametrised) viewshed
            # ==========================================================================
            r_temp_viewshed = #TODO make this a temporary file

            # if considering exposure source dimensions - parametrised cumulative viewshed
            if flags['g']:
                grass.module('r.viewshed.exposure',
                             dsm = R_DSM,
                             output = r_temp_viewshed,
                             sampling_points = v_sampling_points,
                             observer_elevation = V_ELEVATION,
                             max_distance = max_dist_inp,
                             approach = approach,
                             b1_distance = B_1,
                             refraction_coeff = REFR_COEFF,
                             memory = MEMORY,
                             cores = 1, # TODO I think using 1 core is best, since we'll parallelise over this loop
                             flags = FLAGSTRING)

            # else binary cumulative viewshed and then parametrisation
            else:
                # compute cummulative binary viewshed
                r_temp_viewshed_1 = #TODO make this a temporary file
                grass.module('r.viewshed.exposure',
                             dsm = R_DSM,
                             output = r_temp_viewshed_1,
                             sampling_points = v_sampling_points,
                             observer_elevation = V_ELEVATION,
                             max_distance = max_dist_inp,
                             approach = 'binary',
                             b1_distance = B_1,
                             refraction_coeff = REFR_COEFF,
                             memory = MEMORY,
                             cores = 1, # TODO I think using 1 core is best, since we'll parallelise over this loop
                             flags = FLAGSTRING)

                # convert to 0/1
                r_temp_viewshed_2 = #TODO make this a temporary file
                expr = '$outmap = if($inmap > 1, 1, $inmap)'
                grass.mapcalc(
                    expr,
                    inmap=r_temp_viewshed_1,
                    outmap=r_temp_viewshed_2,
                    overwrite=True
                )

                # parametrise
                if approch == 'distance_decay':
                    # TODO implement
                    r_temp_viewshed = ...
                elif approach = 'fuzzy viewshed':
                    # TODO implement
                    r_temp_viewshed = ...
                else:
                    r_temp_viewshed = r_temp_viewshed_2

            # ==========================================================================
            # Multiply by weights map
            # ==========================================================================
            if use_weights:
                r_temp_viewshed = r_temp_viewshed*r_weights

            # ==========================================================================
            # Summarise and store in attribute table
            # ==========================================================================
            univar = grass.read_command(
                        'r.univar',
                        map=r_temp_viewshed
                    )
            sum = int(univar.split('\n')[14].split(':')[1])
            #TODO how to write into an attribute?
            v_source.attrs[attr_vi] = sum

        counter += 1

    end_2 = time.time()
    grass.verbose(_('...finished in {} s'.format(end_2 - start_2)))

    # ## Remove temporary raster - binary viewshed
    # grass.run_command(
    #     'g.remove',
    #     quiet=True,
    #     flags='f',
    #     type='raster',
    #     name=R_VIEWSHED
    # )

    ## Restore original computational region
    # gsl sets region for gsl tasks
    grass.del_temp_region()

    # pygrass sets region for pygrass tasks
    REG.read()
    REG.set_current()
    REG.set_raster_region()

    # ## Convert numpy array of cummulative viewshed to raster
    # numpy2raster(NP_CUM, mtype='FCELL', rastname=r_output, overwrite=True)

    ## Close vector access
    v_sources.close()

    return


if __name__ == '__main__':
    options, flags = grass.parser()
    sys.exit(main())

    # # remove temporary rasters
    # atexit.register(cleanup)
