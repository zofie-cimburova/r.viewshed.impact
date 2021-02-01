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


def do_it_all(t_glob):
    """Conduct weighted and parametrised partial viewshed and cummulate it with
    the previous partial viewsheds
    :param T_gloc: Array of target point coordinates in global coordinate system
    :type t_glob: ndarray
    :param R_DSM: Name of digital surface model raster
    :type R_DSM: string
    :param V_ELEVATION: Observer elevation above the ground
    :type V_ELEVATION: float
    :param NSRES: Cell resolution in N-S direction
    :type NSRES: float
    :param EWRES: Cell resolution in E-W direction
    :type EWRES: float
    :param MAX_DIST: Maximum viewshed distance
    :type MAX_DIST: float
    :param GL_REG_N: North coordinate of global region
    :type GL_REG_N: float
    :param GL_REG_S: South coordinate of global region
    :type GL_REG_S: float
    :param GL_REG_E: East coordinate of global region
    :type GL_REG_E: float
    :param GL_REG_W: West coordinate of global region
    :type GL_REG_W: float
    :param FLAGSTRING: String of flags for r.viewshed
    :type FLAGSTRING: string
    :param R_VIEWSHED: Constant name of binary viewshed
    :type R_VIEWSHED: string
    :param REFR_COEFF: Refraction coefficient
    :type REFR_COEFF: float
    :param MEMORY: Amount of memory to use
    :type MEMORY: int
    :return: 2D array of weighted parametrised cummulative viewshed
    :rtype: ndarray
    """
    # ==========================================================================
    # 1. Set local computational region: +/- MAX_DIST from target point
    # ==========================================================================
    # ensure that local region doesn't exceed global region
    loc_reg_n = min(t_glob[1] + MAX_DIST + NSRES / 2, GL_REG_N)
    loc_reg_s = max(t_glob[1] - MAX_DIST - NSRES / 2, GL_REG_S)
    loc_reg_e = min(t_glob[0] + MAX_DIST + EWRES / 2, GL_REG_E)
    loc_reg_w = max(t_glob[0] - MAX_DIST - EWRES / 2, GL_REG_W)

    # pygrass sets region for pygrass tasks
    bbox = Bbox(loc_reg_n, loc_reg_s, loc_reg_e, loc_reg_w)
    REG.set_bbox(bbox)
    REG.set_raster_region()

    # gsl sets region for gsl tasks
    grass.run_command(
        'g.region',
        n=loc_reg_n,
        s=loc_reg_s,
        e=loc_reg_e,
        w=loc_reg_w
    )

    lreg_shape = [REG.rows, REG.cols]

    # ==========================================================================
    # 2. Calculate binary viewshed and convert to numpy
    # ==========================================================================
    grass.run_command(
        'r.viewshed',
        flags='b' + FLAGSTRING,
        input=R_DSM,
        output=R_VIEWSHED,
        coordinates='{},{}'.format(t_glob[0], t_glob[1]),
        observer_elevation=0.0,
        target_elevation=V_ELEVATION,
        max_distance=MAX_DIST,
        refraction_coeff=REFR_COEFF,
        memory=MEMORY,
        quiet=True,
        overwrite=True
    )

    np_viewshed = raster2numpy(R_VIEWSHED)

    # ==========================================================================
    # 3. Prepare local coordinates and attributes of target point T
    # ==========================================================================
    # Calculate how much of rows/cols of local region lies outside global region
    o_1 = [
        max(t_glob[1] + MAX_DIST + NSRES / 2 - GL_REG_N, 0),
        max(GL_REG_W - (t_glob[0] - MAX_DIST - EWRES / 2), 0)
    ]

    t_loc = np.append(
        np.array(
            [
                MAX_DIST / NSRES + 0.5 - o_1[0] / NSRES,
                MAX_DIST / EWRES + 0.5 - o_1[1] / EWRES
            ]
        ),
        t_glob[2:]
    )

    # ==========================================================================
    # 4. Parametrise viewshed
    # ==========================================================================
    np_viewshed_param = PARAMETRISE_VIEWSHED(lreg_shape, t_loc, np_viewshed)

    # ==========================================================================
    # 5. Cummulate viewsheds
    # ==========================================================================
    ## Determine position of local parametrised viewshed within
    # global cummulative viewshed
    o_2 = [
        int(round((GL_REG_N - loc_reg_n) / NSRES)),  # NS (rows)
        int(round((loc_reg_w - GL_REG_W) / EWRES)),  # EW (cols)
    ]

    ## Add local parametrised viewshed to global cummulative viewshed
    # replace nans with 0 in processed regions
    NP_CUM[o_2[0] : o_2[0] + lreg_shape[0], o_2[1] : o_2[1] + lreg_shape[1]] = np.nan_to_num(
        NP_CUM[o_2[0] : o_2[0] + lreg_shape[0], o_2[1] : o_2[1] + lreg_shape[1]]
    )

    NP_CUM[o_2[0] : o_2[0] + lreg_shape[0], o_2[1] : o_2[1] + lreg_shape[1]] += np_viewshed_param

    ## Clean /.tmp dataset
    # has to be cleaned in loop, otherwise takes time at the end of session
    clean_temp()

    return NP_CUM


def binary(lreg_shape, t_loc, np_viewshed):
    """Weight binary viewshed by constant weight
    :param lreg_shape: Dimensions of local computational region
    :type lreg_shape: list
    :param t_loc: Array of target point coordinates in local coordinate system
    :type t_loc: ndarray
    :param np_viewshed: 2D array of binary viewshed
    :type np_viewshed: ndarray
    :return: 2D array of weighted viewshed
    :rtype: ndarray
    """
    weight = t_loc[-1]
    np_viewshed_param = np_viewshed * weight

    return np_viewshed_param


def solid_angle_reverse(lreg_shape, t_loc, np_viewshed):
    """Calculate solid angle from viewpoints to target based on
    Domingo-Santos et al. (2011) and use it to parametrise binary viewshed
    :param lreg_shape: Dimensions of local computational region
    :type lreg_shape: list
    :param t_loc: Array of target point coordinates in local coordinate system
    :type t_loc: ndarray
    :param np_viewshed: 2D array of binary viewshed
    :type np_viewshed: ndarray
    :param R_DSM: Name of digital surface model raster
    :type R_DSM: string
    :param V_ELEVATION: Observer elevation above the ground
    :type V_ELEVATION: float
    :param NSRES: Cell resolution in N-S direction
    :type NSRES: float
    :param EWRES: Cell resolution in E-W direction
    :type EWRES: float
    :return: 2D array of weighted parametrised viewshed
    :rtype: ndarray
    """
    # 1. Convert DSM to numpy
    np_dsm = raster2numpy(R_DSM)

    # Ensure that values are represented as float (in case of CELL
    # data type) and replace integer NaN with numpy NaN
    dsm_dtype = grass.parse_command('r.info',
                                    map=R_DSM,
                                    flags='g'
                                    )['datatype']
    if dsm_dtype == 'CELL':
        np_dsm = np_dsm.astype(np.float32)
        np_dsm[np_dsm == -2147483648] = np.nan

    # 2. local row, col coordinates and global Z coordinate of observer points V
    #    3D array (lreg_shape[0] x lreg_shape[1] x 3)
    v_loc = np.array(
        [
            np.tile(
                np.arange(0.5, lreg_shape[0] + 0.5).reshape(-1, 1), (1, lreg_shape[1])),
            np.tile(
                np.arange(0.5, lreg_shape[1] + 0.5).reshape(-1, 1).transpose(), (lreg_shape[0], 1)
            ),
            np_dsm + V_ELEVATION,
        ]
    )

    # 3. local row, col coordinates and global Z coordinate of points A, B, C, D
    #    1D array [row, col, Z]
    a_loc = np.array([t_loc[0] + 0.5, t_loc[1] - 0.5, t_loc[3]])

    b_loc = np.array([t_loc[0] - 0.5, t_loc[1] - 0.5, t_loc[4]])

    c_loc = np.array([t_loc[0] - 0.5, t_loc[1] + 0.5, t_loc[5]])

    d_loc = np.array([t_loc[0] + 0.5, t_loc[1] + 0.5, t_loc[6]])

    # 4. vectors a, b, c, d, adjusted for cell size
    #    3D array (lreg_shape[0] x lreg_shape[1] x 3)
    a_vect = np.array(
        [
            (v_loc[0] - a_loc[0]) * NSRES,
            (v_loc[1] - a_loc[1]) * EWRES,
            (v_loc[2] - a_loc[2])
        ]
    )

    b_vect = np.array(
        [
            (v_loc[0] - b_loc[0]) * NSRES,
            (v_loc[1] - b_loc[1]) * EWRES,
            (v_loc[2] - b_loc[2])
        ]
    )

    c_vect = np.array(
        [
            (v_loc[0] - c_loc[0]) * NSRES,
            (v_loc[1] - c_loc[1]) * EWRES,
            (v_loc[2] - c_loc[2])
        ]
    )

    d_vect = np.array(
        [
            (v_loc[0] - d_loc[0]) * NSRES,
            (v_loc[1] - d_loc[1]) * EWRES,
            (v_loc[2] - d_loc[2])
        ]
    )

    # 5. sizes of vectors a, b, c, d
    #    2D array (lreg_shape[0] x lreg_shape[1])
    a_scal = np.sqrt(a_vect[0] ** 2 + a_vect[1] ** 2 + a_vect[2] ** 2)
    b_scal = np.sqrt(b_vect[0] ** 2 + b_vect[1] ** 2 + b_vect[2] ** 2)
    c_scal = np.sqrt(c_vect[0] ** 2 + c_vect[1] ** 2 + c_vect[2] ** 2)
    d_scal = np.sqrt(d_vect[0] ** 2 + d_vect[1] ** 2 + d_vect[2] ** 2)

    # 6. scalar products ab, ac, bc, ad, dc
    #    2D arrays (lreg_shape[0] x lreg_shape[1])
    ab_scal = sum(a_vect * b_vect)
    ac_scal = sum(a_vect * c_vect)
    bc_scal = sum(b_vect * c_vect)
    ad_scal = sum(a_vect * d_vect)
    dc_scal = sum(d_vect * c_vect)

    # 7. determinants of matrix abc, adc
    #    2D arrays (lreg_shape[0] x lreg_shape[1])
    det_abc = (
        a_vect[0] * (b_vect[1] * c_vect[2] - b_vect[2] * c_vect[1])
        - b_vect[0] * (a_vect[1] * c_vect[2] - a_vect[2] * c_vect[1])
        + c_vect[0] * (a_vect[1] * b_vect[2] - a_vect[2] * b_vect[1])
    )

    det_adc = (
        a_vect[0] * (d_vect[1] * c_vect[2] - d_vect[2] * c_vect[1])
        - d_vect[0] * (a_vect[1] * c_vect[2] - a_vect[2] * c_vect[1])
        + c_vect[0] * (a_vect[1] * d_vect[2] - a_vect[2] * d_vect[1])
    )

    # 8. solid angle
    solid_angle_1 = np.arctan2(
        det_abc, a_scal * b_scal * c_scal + ab_scal * c_scal + ac_scal * b_scal + bc_scal * a_scal
    )

    solid_angle_2 = np.arctan2(
        det_adc, a_scal * d_scal * c_scal + ad_scal * c_scal + ac_scal * d_scal + dc_scal * a_scal
    )

    solid_angle = np.absolute(solid_angle_1) + np.absolute(solid_angle_2)

    # 9. Multiply solid angle by binary viewshed and weight
    weight = t_loc[-1]
    np_viewshed_param = solid_angle * np_viewshed * weight

    return np_viewshed_param


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


def vertical_angle_reverse(lreg_shape, t_loc, np_viewshed):
    """Calculate vertical angle from viewpoints to target based on
    Chamberlain (2011) and Chamberlain & Meither (2013) and use it to
    parametrise binary viewshed
    :param lreg_shape: Dimensions of local computational region
    :type lreg_shape: list
    :param t_loc: Array of target point coordinates in local coordinate system
    :type t_loc: ndarray
    :param np_viewshed: 2D array of binary viewshed
    :type np_viewshed: ndarray
    :param R_DSM: Name of digital surface model raster
    :type R_DSM: string
    :param V_ELEVATION: Observer elevation above the ground
    :type V_ELEVATION: float
    :param NSRES: Cell resolution in N-S direction
    :type NSRES: float
    :param EWRES: Cell resolution in E-W direction
    :type EWRES: float
    :return: 2D array of weighted parametrised viewshed
    :rtype: ndarray
    """
    # 1. Convert DSM to numpy
    np_dsm = raster2numpy(R_DSM)

    # Ensure that values are represented as float (in case of CELL
    # data type) and replace integer NaN with numpy NaN
    dsm_dtype = grass.parse_command(
                    'r.info',
                    map=R_DSM,
                    flags='g'
                    )['datatype']
    if dsm_dtype == 'CELL':
        np_dsm = np_dsm.astype(np.float32)
        np_dsm[np_dsm == -2147483648] = np.nan

    # 2. local row, col coordinates and global Z coordinate of observer points V
    #    3D array (lreg_shape[0] x lreg_shape[1] x 3)
    v_loc = np.array(
        [
            np.tile(
                np.arange(0.5, lreg_shape[0] + 0.5).reshape(-1, 1), (1, lreg_shape[1])
            ),
            np.tile(
                np.arange(0.5, lreg_shape[1] + 0.5).reshape(-1, 1).transpose(), (lreg_shape[0], 1)
            ),
            np_dsm + V_ELEVATION,
        ]
    )

    # 3. vector VT, adjusted for cell size
    #    3D array (lreg_shape[0] x lreg_shape[1] x 3)
    v_vect = np.array(
        [
            (v_loc[0] - t_loc[0]) * NSRES,
            (v_loc[1] - t_loc[1]) * EWRES,
            (v_loc[2] - t_loc[2])
        ]
    )

    # 4. projection of vector VT to XZ and YZ plane, adjusted for cell size
    v_vect_ns = np.array([(v_loc[0] - t_loc[0]) * NSRES, v_loc[2] - t_loc[2]])
    v_vect_ew = np.array([(v_loc[1] - t_loc[1]) * EWRES, v_loc[2] - t_loc[2]])

    v_vect_ns_unit = v_vect_ns / np.linalg.norm(v_vect_ns, axis=0)
    v_vect_ew_unit = v_vect_ew / np.linalg.norm(v_vect_ew, axis=0)

    # 5. size of vector VT
    #    2D array (lreg_shape[0] x lreg_shape[1])
    v_scal = np.sqrt(v_vect[0] ** 2 + v_vect[1] ** 2 + v_vect[2] ** 2)

    # 6. vector n, its projection to XZ, YZ plane
    #   1D array [X, Z], [Y, Z]
    n_vect_ns = [1, -t_loc[4]]
    n_vect_ew = [1, t_loc[3]]

    n_vect_ns_unit = n_vect_ns / np.linalg.norm(n_vect_ns, axis=0)
    n_vect_ew_unit = n_vect_ew / np.linalg.norm(n_vect_ew, axis=0)

    # 7. angles beta (ns), theta (ew) (0-90 degrees)
    #    2D array (lreg_shape[0] x lreg_shape[1])
    beta = np.arccos(
        n_vect_ns_unit[0] * v_vect_ns_unit[:][0] + n_vect_ns_unit[1] * v_vect_ns_unit[:][1]
    )
    beta = np.where(beta > math.pi / 2, beta - math.pi / 2, beta)

    theta = np.arccos(
        n_vect_ew_unit[0] * v_vect_ew_unit[:][0] + n_vect_ew_unit[1] * v_vect_ew_unit[:][1]
    )
    theta = np.where(theta > math.pi / 2, theta - math.pi / 2, theta)

    # 8. vertical angle adjusted for distance weight
    vertical_angle = np.cos(beta) * np.cos(theta) * ((NSRES * EWRES) / v_scal ** 2)

    # 9. Multiply vertical angle by binary viewshed and weight
    weight = t_loc[-1]
    np_viewshed_param = vertical_angle * np_viewshed * weight

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
    v_sources.open('r')
    no_sources = v_sources.num_primitive_of('area')

    ## Attribute to store visual impact values
    attr_vi = options['attribute']

    if attr_vi in v_sources[1].attrs.keys(): # TODO how to check if attribute already exists?
        grass.fatal('Attribute <%s> already exists' % attr_vi)
    else:
        grass.run_command(
            'v.db.addcolumn',
            map=sources,
            columns='{} {}'.format(attr_vi, 'double'))

    ## Weights
    r_weights = options['weight']


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

    # # ==========================================================================
    # # Random sample exposure source with source points T
    # # ==========================================================================
    # # TODO if v_sources is points map - use instead of sampling?
    # # if v_sources:
    # #     # go for using input vector map as sampling points
    # #     v_sources_sample = v_sources
    # #     grass.verbose("Using sampling points from input vector map")
    #
    # # else:
    # ## go for sampling
    #
    # univar = grass.read_command(
    #             'r.univar',
    #             map=r_source
    #         )
    # source_ncells = int(univar.split('\n')[5].split(':')[1])
    #
    # # number of cells in sample
    # sample_ncells = int(float(source_sample_density) * source_ncells / 100)
    # grass.verbose("{} source points".format(source_ncells))
    #
    # if sample_ncells == 0:
    #     grass.fatal('The analysis cannot be conducted for 0 sampling points.')
    # else:
    #     grass.verbose("Distributing {} sampling points".format(sample_ncells))
    #
    # # min. distance between samples set to region resolution
    # sample_distance = NSRES
    #
    # v_sources_sample = sample_raster_with_points(
    #     r_source, source_cat, sample_ncells, sample_distance, 'tmp_rand_pts_vect', seed
    # )
    # TMP_VECT.append(v_sources_sample)

    # ==========================================================================
    # Prepare maps for viewshed parametrisation
    # ==========================================================================
    ## Prepare a list of maps to extract attributes from
    # DSM values
    attr_map_list = [R_DSM]

    ## Precompute values A, B, C, D for solid angle approach
    # using moving window [row, col]
    if approach == 'solid_angle':

        r_a_z = 'tmp_A_z'
        r_b_z = 'tmp_B_z'
        r_c_z = 'tmp_C_z'
        r_d_z = 'tmp_D_z'

        expr = ';'.join(
            [
            '$outmap_A = (if(isnull($inmap[0,0]), 0, $inmap[0, 0]) + \
                          if(isnull($inmap[0,-1]), 0, $inmap[0, -1]) + \
                          if(isnull($inmap[1, -1]), 0, $inmap[1, -1]) + \
                          if(isnull($inmap[1, 0]), 0, $inmap[1, 0])) / \
                          (!isnull($inmap[0, 0]) + !isnull($inmap[0, -1]) + \
                          !isnull($inmap[1, -1]) + !isnull($inmap[1, 0]))',
            '$outmap_B = (if(isnull($inmap[-1, 0]), 0, $inmap[-1, 0]) + \
                          if(isnull($inmap[-1, -1]), 0, $inmap[-1, -1]) + \
                          if(isnull($inmap[0, -1]), 0, $inmap[0, -1]) + \
                          if(isnull($inmap[0, 0]), 0, $inmap[0, 0])) / \
                          (!isnull($inmap[-1, 0]) + !isnull($inmap[-1, -1]) + \
                          !isnull($inmap[0, -1]) + !isnull($inmap[0, 0]))',
            '$outmap_C = (if(isnull($inmap[-1, 1]), 0, $inmap[-1, 1]) + \
                          if(isnull($inmap[-1, 0]), 0, $inmap[-1, 0]) + \
                          if(isnull($inmap[0, 0]), 0, $inmap[0, 0]) + \
                          if(isnull($inmap[0, 1]), 0, $inmap[0, 1])) / \
                          (!isnull($inmap[-1, 1]) + !isnull($inmap[-1, 0]) + \
                          !isnull($inmap[0, 0]) + !isnull($inmap[0, 1]))',
            '$outmap_D = (if(isnull($inmap[0, 1]), 0, $inmap[0, 1]) + \
                          if(isnull($inmap[0, 0]), 0, $inmap[0, 0]) + \
                          if(isnull($inmap[1, 0]), 0, $inmap[1, 0]) + \
                          if(isnull($inmap[1, 1]), 0, $inmap[1, 1])) / \
                          (!isnull($inmap[0, 1]) + !isnull($inmap[0, 0]) + \
                          !isnull($inmap[1, 0]) + !isnull($inmap[1, 1]))'
          ]
        )
        grass.mapcalc(
            expr,
            inmap=R_DSM,
            outmap_A=r_a_z,
            outmap_B=r_b_z,
            outmap_C=r_c_z,
            outmap_D=r_d_z,
            overwrite=True
        )

        attr_map_list.extend([r_a_z, r_b_z, r_c_z, r_d_z])
        TMP_RAST.extend([r_a_z, r_b_z, r_c_z, r_d_z])

    # Precompute values dz/dx (e-w direction), dz/dy (n-s direction)
    # using moving window [row, col]
    # TODO2 how to deal with remaining edge effect in slope computation?
    elif approach == 'vertical_angle':

        r_dz_dew = 'tmp_dz_dew'
        r_dz_dns = 'tmp_dz_dns'

        expr = ';'.join(
            [
            '$outmap_ew = (sqrt(2) * if(isnull($inmap[-1, 1]), 0, $inmap[-1, 1]) + \
                          2 * if(isnull($inmap[0, 1]), 0, $inmap[0, 1]) + \
                          sqrt(2) * if(isnull($inmap[1, 1]), 0, $inmap[1, 1]) - \
                          sqrt(2) * if(isnull($inmap[-1, -1]), 0, $inmap[-1, -1]) - \
                          2 * if(isnull($inmap[0, -1]), 0, $inmap[0, -1]) - \
                          sqrt(2) * if(isnull($inmap[1, -1]), 0, $inmap[1, -1])) / \
                          ((if(isnull($inmap[-1, 1]), 0, 1) + \
                           if(isnull($inmap[0, 1]), 0, 2) + \
                           if(isnull($inmap[1, 1]), 0, 1) + \
                           if(isnull($inmap[-1, -1]), 0, 1) + \
                           if(isnull($inmap[0, -1]), 0, 2) + \
                           if(isnull($inmap[1, -1]), 0, 1)) * $w_ew)',
            '$outmap_ns = (sqrt(2) * if(isnull($inmap[-1, -1]), 0, $inmap[-1, -1]) + \
                          2 * if(isnull($inmap[-1, 0]), 0, $inmap[-1, 0]) + \
                          sqrt(2) * if(isnull($inmap[-1, 1]), 0, $inmap[-1, 1]) - \
                          sqrt(2) * if(isnull($inmap[1, -1]), 0, $inmap[1, -1]) - \
                          2 * if(isnull($inmap[1, 0]), 0, $inmap[1, 0]) - \
                          sqrt(2) * if(isnull($inmap[1, 1]), 0, $inmap[1, 1])) / \
                          ((if(isnull($inmap[-1, -1]), 0, 1) + \
                           if(isnull($inmap[-1, 0]), 0, 2) + \
                           if(isnull($inmap[-1, 1]), 0, 1) + \
                           if(isnull($inmap[1, -1]), 0, 1) + \
                           if(isnull($inmap[1, 0]), 0, 2) + \
                           if(isnull($inmap[1, 1]), 0, 1)) * $w_ns)'
            ]
        )

        grass.mapcalc(
            expr,
            inmap=R_DSM,
            outmap_ew=r_dz_dew,
            outmap_ns=r_dz_dns,
            w_ew=EWRES,
            w_ns=NSRES,
            overwrite=True
        )

        attr_map_list.extend([r_dz_dew, r_dz_dns])
        TMP_RAST.extend([r_dz_dew, r_dz_dns])

    # ## Use viewshed weights if provided
    # if r_weights:
    #     attr_map_list.append(r_weights)

    # ## Extract attribute values
    # target_pts_grass = grass.read_command(
    #     'r.what',
    #     flags='v',
    #     map=attr_map_list,
    #     points=v_sources_sample,
    #     separator='|',
    #     null_value='*'
    # )

    # columns to use depending on parametrisation method
    usecols = list(range(0, 4 + len(attr_map_list)))
    usecols.remove(3) # skip 3rd column - site_name

    # # convert coordinates and attributes of target points T to numpy array
    # target_pts_np = txt2numpy(
    #     target_pts_grass, sep='|', names=None, null_value='*', usecols=usecols, structured=False
    # )

    # # if one point only - 0D array which cannot be used in iteration
    # if target_pts_np.ndim == 1:
    #     target_pts_np = target_pts_np.reshape(1, -1)
    # no_points = target_pts_np.shape[0]
    #
    # # if viewshed weights not set by flag - set weight to 1 for all pts
    # if not r_weights:
    #     weights_np = np.ones((no_points, 1))
    #     target_pts_np = np.hstack((target_pts_np, weights_np))
    #
    # grass.message("target_pts_np: {}".format(target_pts_np))

    # ==========================================================================
    # Iterate over exposure source polygons
    # and calculate their visual impact
    # using weighted cumulative parametrised viewshed
    # ==========================================================================
    # counter to print progress in percentage
    counter = 0

    # # 2D numpy array to store the partial cummulative viewsheds
    # global NP_CUM
    # NP_CUM = np.empty((gl_reg_rows, gl_reg_cols), dtype=np.single)
    # NP_CUM[:] = np.nan

    # # random name of binary viewshed
    # global R_VIEWSHED
    # R_VIEWSHED = grass.tempname(6)
    # TMP_RAST.append(R_VIEWSHED)

    # parametrisation function
    global PARAMETRISE_VIEWSHED
    if approach == 'solid_angle':
        PARAMETRISE_VIEWSHED = solid_angle_reverse

    elif approach == 'distance_decay':
        PARAMETRISE_VIEWSHED = distance_decay_reverse

    elif approach == 'fuzzy_viewshed':
        PARAMETRISE_VIEWSHED = fuzzy_viewshed_reverse

    elif approach == 'vertical_angle':
        PARAMETRISE_VIEWSHED = vertical_angle_reverse

    else:
        PARAMETRISE_VIEWSHED = binary

    start_2 = time.time()
    grass.verbose(_('Iterating over trees...'))

    for v_source in v_sources.viter('areas'):
        ## Display a progress info message
        grass.percent(counter, no_sources, 1)

        ## Only process features which have attribute table
        if v_source.attrs is None:
            grass.verbose("Problem") #TODO what are the features without attributes?

        else:
            source_id = v_source.attrs['CrownID']
            grass.verbose('Processing exposure source ID: {}'.format(source_id))

            ## Create random points
            # TODO how to do that? I.e. what module?
            # TODO when to do that? I.e. for each tree separately, or at the beginning once for all trees?

            ## Calculate parametrised viewshed
            # TODO reuse functions from r.viewshed.exposure - or insert finished module?

            ## Multiply by weights
            # TODO

            ## Summarise
            # TODO

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
