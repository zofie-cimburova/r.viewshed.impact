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
#python3 /home/NINA.NO/zofie.cimburova/PhD/Paper4/SRC/r.viewshed.impact.py

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
#% required: yes
#% description: Name of attribute column to store visual impact values
#%end

#%option G_OPT_R_INPUT
#% key: weight
#% required: no
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
#% key: range
#% type: double
#% required: no
#% key_desc: value
#% options: 0.0- , -1 for infinity
#% description: Exposure range
#% answer: 100
#% guisection: Viewshed settings
#%end

#%option
#% key: function
#% type: string
#% required: no
#% options: binary, distance_decay, fuzzy_viewshed, visual_magnitude, solid_angle
#% key_desc: name
#% description: Viewshed parametrisation function
#% guisection: Viewshed settings
#% answer: binary
#%end

#%option
#% key: b1_distance
#% type: double
#% required: no
#% key_desc: value
#% description: Radius around the viewpoint where clarity is perfect. Used in fuzzy viewshed function.
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
#% description: Number of cores to use in paralellization
#% answer: 1
#%end

import os
import math
import time
import atexit
import sys
import subprocess
import numpy as np
import csv


from grass.pygrass.raster import raster2numpy
from grass.pygrass.raster import numpy2raster

from grass.pygrass.gis.region import Region
from grass.pygrass.vector.basic import Bbox
from grass.pygrass.vector import VectorTopo
from grass.pygrass.modules import Module
from grass.pygrass.raster import RasterRow
from grass.pygrass.gis import Mapset


import grass.script as grass
from grass.script import utils as grassutils

# enable coordinate systems with various axis orientation (now assuming Y-north, X-east)
#      > TODO2 is it so that X is always to the east and Y to the north? I've checked with e.g. s-jtsk and it seems so
#       > Stefan asks


# global variables
TEMPNAME = grass.tempname(12)

def cleanup():
    """Remove raster and vector maps stored in a list"""
    grass.run_command(
        "g.remove",
        flags="f",
        type="raster,vector,region",
        pattern="{}_*".format(TEMPNAME),
        quiet=True,
        stderr=subprocess.PIPE,
    )

    # Reset mask if user MASK was present
    if (
        RasterRow("MASK", Mapset().name).exist()
        and RasterRow("MASK_{}".format(TEMPNAME)).exist()
    ):
        grass.run_command("r.mask", flags="r", quiet=True)
    reset_mask()


def unset_mask():
    """Deactivate user mask"""
    if RasterRow("MASK", Mapset().name).exist():
        try:
            grass.run_command(
                "g.copy",
                quiet=True,
                raster="MASK,MASK_{}".format(TEMPNAME),
                stderr=subprocess.DEVNULL,
            )
            grass.run_command(
                "g.remove",
                quiet=True,
                type="raster",
                name="MASK",
                stderr=subprocess.DEVNULL,
                flags="f",
            )
        except:
            pass


def reset_mask():
    """Re-activate user mask"""
    if RasterRow("MASK_{}".format(TEMPNAME)).exist():
        try:
            grass.warning("reset mask")
            grass.run_command(
                "g.copy",
                quiet=True,
                raster="MASK_{},MASK".format(TEMPNAME),
                stderr=subprocess.DEVNULL,
            )
            grass.run_command(
                "g.remove",
                quiet=True,
                type="raster",
                name="MASK_{}".format(TEMPNAME),
                stderr=subprocess.DEVNULL,
                flags="f",
            )
        except:
            pass


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
    :param nsres: Cell resolution in N-S direction
    :type nsres: float
    :param ewres: Cell resolution in E-W direction
    :type ewres: float
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
            (v_loc[0] - t_loc[0]) * nsres,
            (v_loc[1] - t_loc[1]) * ewres
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
    :param b_1: Radius of zone around the viewpoint where clarity is perfect
    :type b_1: float
    :param max_dist: Radius of crossover point (maximum viewshed distance)
    :type max_dist: float
    :param nsres: Cell resolution in N-S direction
    :type nsres: float
    :param ewres: Cell resolution in E-W direction
    :type ewres: float
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
            (v_loc[0] - t_loc[0]) * nsres,
            (v_loc[1] - t_loc[1]) * ewres
        ]
    )

    # 3. size of vector VT
    #    2D array (lreg_shape[0] x lreg_shape[1])
    v_scal = np.sqrt(v_vect[0] ** 2 + v_vect[1] ** 2)

    # replace 0 distance for central pixel by 1 to avoid division by 0
    v_scal = np.where(v_scal == 0, 1, v_scal)

    # 4. fuzzy viewshed function
    fuzzy_viewshed = np.where(
        v_scal <= b_1, 1, 1 / (1 + ((v_scal - b_1) / max_dist) ** 2)
        )

    # 5. Multiply fuzzy viewshed by binary viewshed and weight
    weight = t_loc[-1]
    np_viewshed_param = fuzzy_viewshed * np_viewshed * weight

    return np_viewshed_param


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
    r_dsm = options['dsm']
    gfile_dsm = grass.find_file(name=r_dsm, element='cell')
    if not gfile_dsm['file']:
        grass.fatal('Raster map <%s> not found' % r_dsm)

    ## Exposure source
    v_sources = options['exposure_source'].split("@")[0]
    #TODO why can only vector map in current mapset be used?
    gfile_source = grass.find_file(name=v_sources, element='vector')
    if not gfile_source['file']:
        grass.fatal('Vector map <%s> not found' % v_sources)

    # build topology in case it got corrupted
    grass.run_command('v.build',
                      map=v_sources,
                      quiet=True)

    ## Weights
    r_weights = options['weight']

    use_weights = 0
    if r_weights:
        use_weights = 1
        gfile_weights = grass.find_file(name=r_weights, element='cell')
        if not gfile_weights['file']:
            grass.fatal('Raster map <%s> not found' % r_weights)

    ## Attribute to store visual impact values
    a_impact = options['attribute']

    #TODO how to check better if attribute already exists and what to do if it exists?
    # if a_impact in v_sources_topo[1].attrs.keys():
    #     grass.fatal('Attribute <%s> already exists' % a_impact)
    # else:
    #     grass.run_command(
    #         'v.db.addcolumn',
    #         map=v_sources,
    #         columns='{} double precision'.format(a_impact))

    # TODO how to write results to attribute table? Now written to file
    t_result = "/home/NINA.NO/zofie.cimburova/PhD/Paper4/DATA/tmp_out.csv"

    ## Viewshed settings
    flagstring = ''
    if flags['r']:
        flagstring += 'r'
    if flags['c']:
        flagstring += 'c'

    v_elevation = float(options['observer_elevation'])
    range = float(options['range'])
    function = options['function']
    b_1 = float(options['b1_distance'])
    refr_coeff = float(options['refraction_coeff'])

    # test values
    if v_elevation < 0.0:
        grass.fatal('Observer elevation must be larger than or equal to 0.0.')
    if range <= 0.0 and range != -1:
        grass.fatal('Maximum visibility radius must be larger than 0.0.')
    if function == 'fuzzy_viewshed' and range == -1:
        grass.fatal('Maximum visibility radius cannot be infinity for fuzzy viewshed approch.')
    if function == 'fuzzy_viewshed' and b_1 > range:
        grass.fatal(
            'Maximum visibility radius must be larger than radius around the viewpoint where clarity is perfect.'
        )

    ## Sampling settings
    source_sample_density = float(options['sample_density'])
    seed = options['seed']

    if not seed: # if seed is not set, set it to process number
        seed = os.getpid()

    ## Optional
    memory = int(options['memory'])
    cores = int(options['cores'])


    # ==========================================================================
    # Region and mask settings
    # ==========================================================================
    # check that location is not in lat/long
    if grass.locn_is_latlong():
        grass.fatal('The analysis is not available for lat/long coordinates.')

    user_mask = False
    if RasterRow("MASK", Mapset().name).exist():
        grass.warning(_("Current MASK is temporarily renamed."))
        user_mask = True
        unset_mask()

    # store the current region settings
    # TODO
    # either only grass.script region (grass.run_command(g.region))
    # or environment settings in r.viewshed.exposure (env=c_env)
    # # Create processing environment with region information
    # c_env = os.environ.copy()
    # c_env["GRASS_REGION"] = grass.region_env(
    #     n=reg_n, s=reg_s, e=reg_e, w=reg_w
    # )
    # # grass.use_temp_region()

    # get comp. region parameters
    reg = Region()
    nsres, ewres = reg.nsres, reg.ewres

    # check that nsres equals ewres
    if nsres != ewres:
        grass.fatal('Variable north-south and east-west 2D grid resolution is not supported')


    # ==========================================================================
    # Rasterise vector source map
    # ==========================================================================
    r_sources = "{}_src_rast".format(TEMPNAME)
    grass.run_command('v.to.rast',
                      input=v_sources,
                      output=r_sources,
                      use='cat',
                      overwrite=True,
                      quiet=True)

    # ==========================================================================
    # Iterate over exposure source polygons
    # and calculate their visual impact
    # using weighted cumulative parametrised viewshed
    # ==========================================================================
    sources_list = grass.read_command("r.stats",
                                      flags="cn",
                                      input=r_sources,
                                      separator="|")

    sources_np = txt2numpy(
        sources_list,
        sep="|",
        names=None,
        null_value="*",
        structured=False,
    )

    # counter to print progress in percentage
    counter = 0
    no_sources = sources_np.shape[0]

    with open(t_result, "a") as outfile:
        fieldnames = ['source_cat','value']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for source in sources_np:
            ## Display a progress info message
            grass.percent(counter, no_sources, 1)
            counter += 1

            # TODO why is source_id not int?
            source_id = int(source[0])
            source_ncell = int(source[1])
            grass.verbose('Processing source {}, {}%'.format(source_id, counter/no_sources*100))

            # ==========================================================================
            # Adjust computational region to range around processed exposure source
            # ==========================================================================
            # TODO - how to do this with current raster approach?
            # source_bbox = v_source.bbox()
            #
            # grass.run_command('g.region',
            #                   align=r_dsm,
            #                   n=source_bbox.north + range,
            #                   s=source_bbox.south - range,
            #                   e=source_bbox.east + range,
            #                   w=source_bbox.west - range)

            # ==========================================================================
            # Calculate cummulative (parametrised) viewshed
            # ==========================================================================
            r_temp_viewshed = "{}_viewshed".format(TEMPNAME)

            # if considering exposure source dimensions - parametrised cumulative viewshed
            if flags['g']:
                grass.run_command('r.viewshed.exposure',
                                 dsm = r_dsm,
                                 output = r_temp_viewshed,
                                 source = r_sources,
                                 sourcecat = source_id,
                                 observer_elevation = v_elevation,
                                 range = range,
                                 function = function,
                                 b1_distance = b_1,
                                 sample_density = source_sample_density,
                                 refraction_coeff = refr_coeff,
                                 memory = memory,
                                 cores = cores, # TODO I think using 1 core is best, since we'll parallelise over this loop
                                 flags = flagstring,
                                 quiet=True,
                                 overwrite=True)


            # else binary cumulative viewshed and then parametrisation
            else:
                # compute cummulative binary viewshed
                r_temp_viewshed_1 = "{}_viewshed_binary".format(TEMPNAME)
                grass.run_command('r.viewshed.exposure',
                                 dsm = r_dsm,
                                 output = r_temp_viewshed_1,
                                 source = r_sources,
                                 sourcecat = source_id,
                                 observer_elevation = v_elevation,
                                 range = range,
                                 function = 'binary',
                                 b1_distance = b_1,
                                 sample_density = source_sample_density,
                                 refraction_coeff = refr_coeff,
                                 memory = memory,
                                 cores = cores, # TODO I think using 1 core is best, since we'll parallelise over this loop
                                 flags = flagstring,
                                 quiet=True,
                                 overwrite=True)

                # convert to 0/1
                r_temp_viewshed_2 = "{}_viewshed_threshold".format(TEMPNAME)
                expr = '$outmap = if($inmap > 1, 1, $inmap)'
                grass.mapcalc(
                    expr,
                    inmap=r_temp_viewshed_1,
                    outmap=r_temp_viewshed_2,
                    overwrite=True
                )

                # parametrise
                if function == 'distance_decay':
                    # TODO implement
                    #r_temp_viewshed = ...
                    pass
                elif function == 'fuzzy viewshed':
                    # TODO implement
                    #r_temp_viewshed = ...
                    pass
                else:
                    r_temp_viewshed = r_temp_viewshed_2

            # ==========================================================================
            # Multiply by weights map
            # ==========================================================================
            r_temp_viewshed_weighted = 'tmp_viewshed_weighted_{}'.format(source_id)
            if use_weights:
                grass.mapcalc('$outmap = $map_a * $map_b',
                             map_a=r_temp_viewshed,
                             map_b=r_weights,
                             outmap=r_temp_viewshed_weighted,
                             overwrite=True,
                             quiet=grass.verbosity() <= 1)
            else:
                 r_temp_viewshed_weighted=r_temp_viewshed


            # ==========================================================================
            # Summarise raster values and write to attribute table
            # ==========================================================================
            univar = grass.read_command(
                        'r.univar',
                        map=r_temp_viewshed_weighted
                    )
            if flags['g']:
                # normalise by number of points
                sum = float(univar.split('\n')[14].split(':')[1])/source_ncell
            else:
                sum = float(univar.split('\n')[14].split(':')[1])

            writer.writerow({'source_cat':source_id,
                             'value':sum})

    # ## Restore original computational region
    # # gsl sets region for gsl tasks
    # grass.del_temp_region()
    #
    # # pygrass sets region for pygrass tasks
    # reg.read()
    # reg.set_current()
    # reg.set_raster_region()


    # Remove temporary files and reset mask if needed
    cleanup()


if __name__ == '__main__':
    options, flags = grass.parser()
    atexit.register(cleanup)
    sys.exit(main())
