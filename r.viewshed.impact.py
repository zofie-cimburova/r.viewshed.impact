#!/usr/bin/env python3

"""
MODULE:       r.viewshed.impact

AUTHOR(S):    Zofie Cimburova, Stefan Blumentrath

PURPOSE:      Computes visual impact of defined exposure source using weighted parametrised viewshed analysis

COPYRIGHT:    (C) 2021 by Zofie Cimburova, Stefan Blumentrath, and the GRASS Development Team

REFERENCES:   TODO1 reference papers used and the paper to be published

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

#%option G_OPT_V_MAP
#% key: exposure_source
#% label: Name of input map of exposure source locations
#% required: yes
#% guidependency: range_layer, range_col
#%end

#%option G_OPT_V_FIELD
#% key: range_layer
#% guidependency: range_col
#%end

#%option
#% key: column
#% required: yes
#% label: Name of attribute column to store visual impact values
#%end

#%option G_OPT_R_INPUT
#% key: dsm
#% required: yes
#% label: Name of input digital surface raster map
#%end

#%option G_OPT_R_INPUT
#% key: weight
#% required: no
#% label: Name of input weights raster map
#% guisection: Weights settings
#%end

#%flag
#% key: w
#% label: Keep intermediate viewshed maps
#% guisection: Viewshed settings
#%end

#%flag
#% key: c
#% label: Consider the curvature of the earth (current ellipsoid)
#% guisection: Viewshed settings
#%end

#%option
#% key: observer_elevation
#% type: double
#% required: no
#% key_desc: value
#% label: Observer elevation above the ground
#% description: 0.0-
#% options: 0.0-
#% answer: 1.5
#% guisection: Viewshed settings
#%end

#%option G_OPT_DB_COLUMN
#% key: range_col
#% required: no
#% label: Name of attribute column containing exposure range
#% guisection: Viewshed settings
#%end

#%option
#% key: range_max
#% type: double
#% required: no
#% key_desc: value
#% label: Maximum exposure range
#% description: 0.0- , -1 for infinity
#% options: 0.0-
#% answer: 100
#% guisection: Viewshed settings
#%end

#%option
#% key: function
#% type: string
#% required: no
#% key_desc: name
#% label: Viewshed parametrisation function
#% description: None, Binary, Distance decay, Fuzzy viewshed, Visual magnitude, Solid angle
#% options: None, Binary, Distance decay, Fuzzy viewshed, Visual magnitude, Solid angle
#% answer: Distance decay
#% guisection: Viewshed settings
#%end

#%option
#% key: b1_distance
#% type: double
#% required: no
#% key_desc: value
#% label: Radius around the observer where clarity is perfect. Used in fuzzy viewshed function.
#% guisection: Viewshed settings
#% answer: 10
#%end

#%option
#% key: sample_density
#% type: double
#% required: no
#% key_desc: value
#% label: Density of sampling points
#% options: 0.0-100.0
#% description: 0.0-100.0
#% answer: 25
#% guisection: Sampling settings
#%end

#%option
#% key: seed
#% type: integer
#% required: no
#% key_desc: value
#% label: Random seed, default [random]
#% options: 0-
#% description: 0-
#% guisection: Sampling settings
#%end

#%flag
#% key: r
#% label: Consider the effect of atmospheric refraction
#% guisection: Refraction
#%end

#%option
#% key: refraction_coeff
#% type: double
#% required: no
#% key_desc: value
#% label: Refraction coefficient
#% options: 0.0-1.0
#% description: 0.0-1.0
#% answer: 0.14286
#% guisection: Refraction
#%end

#%option
#% key: memory
#% type: integer
#% required: no
#% key_desc: value
#% label: Amount of memory to use in MB
#% options: 1-
#% description: 1-
#% answer: 500
#%end

#%option
#% key: cores
#% type: integer
#% required: no
#% key_desc: value
#% label: Number of cores to use in parrallelization
#% description: 1-
#% options: 1-
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


# global variables
TEMPNAME = grass.tempname(12)
EXCLUDE = None

def cleanup():
    """Remove raster and vector maps stored in a list"""
    grass.run_command(
        "g.remove",
        flags="f",
        type="raster,vector,region",
        pattern="{}_*".format(TEMPNAME),
        exclude=EXCLUDE,
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

    # convert to pygrass VectorTopo object
    vt_sources = VectorTopo(v_sources)
    vt_sources.open('r')
    no_sources = vt_sources.num_primitive_of('area')

    ## Weights
    r_weights = options['weight']

    if r_weights:
        gfile_weights = grass.find_file(name=r_weights, element='cell')
        if not gfile_weights['file']:
            grass.fatal('Raster map <%s> not found' % r_weights)

    ## Column to store visual impact values
    a_impact = options['column']

    #TODO ISSUE 6: how to check better if attribute already exists and what to do if it exists?
    # if a_impact in v_sources_topo[1].attrs.keys():
    #     grass.fatal('Attribute <%s> already exists' % a_impact)
    # else:
    #     grass.run_command(
    #         'v.db.addcolumn',
    #         map=v_sources,
    #         columns='{} double precision'.format(a_impact))

    # TODO ISSUE 7: how to write results to attribute table? Now written to file
    t_result = "/home/NINA.NO/zofie.cimburova/PhD/Paper4/DATA/tmp_out.csv"

    ## Viewshed settings
    flagstring = ''
    if flags['r']:
        flagstring += 'r'
    if flags['c']:
        flagstring += 'c'

    v_elevation = float(options['observer_elevation'])
    range = float(options['range_max'])
    function = options['function']
    b_1 = float(options['b1_distance'])
    refr_coeff = float(options['refraction_coeff'])

    # test values
    if v_elevation < 0.0:
        grass.fatal('Observer elevation must be larger than or equal to 0.0.')
    if range <= 0.0 and range != -1:
        grass.fatal('Maximum visibility radius must be larger than 0.0.')
    if function == 'Fuzzy viewshed' and range == -1:
        grass.fatal('Maximum visibility radius cannot be infinity for fuzzy viewshed approch.')
    if function == 'Fuzzy viewshed' and b_1 > range:
        grass.fatal(
            'Maximum visibility radius must be larger than radius around the viewpoint where clarity is perfect.'
        )

    # option for binary output instead of cummulative
    binary_output = False
    if function == "None":
        function = "Binary"
        binary_output = True

    ## Sampling settings
    source_sample_density = float(options['sample_density'])
    seed = options['seed']

    # if seed is not set, set it to process number
    if not seed:
        seed = os.getpid()

    ## Optional
    memory = int(options['memory'])
    cores = int(options['cores'])

    ## Keep or delete intermediate map
    global EXCLUDE
    if flags['w']:
        if r_weights:
            EXCLUDE = "*_exposure_weighted"
        else:
            EXCLUDE = "*_exposure"

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
    # TODO ISSUE 9
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
    # Iteration over sources
    # ==========================================================================
    counter = 0

    with open(t_result, "a") as outfile:
        fieldnames = ['src_cat','value']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for vt_source in vt_sources.viter('areas'):
            ## Display progress info message
            grass.percent(counter, no_sources, 1)
            counter += 1

            ## Only process features which have attribute table
            #TODO what are the features without attributes?
            if vt_source.attrs is None:
                grass.verbose("Problem")
            else:
                src_cat = vt_source.attrs['cat']
                grass.verbose('Processing source cat: {}'.format(src_cat))

                # ==============================================================
                # Set computational region to range around processed source
                # ==============================================================
                # TODO how to account for current settings of computational region?
                src_bbox = vt_source.bbox()

                grass.run_command('g.region',
                                  align=r_dsm,
                                  n=src_bbox.north + range,
                                  s=src_bbox.south - range,
                                  e=src_bbox.east + range,
                                  w=src_bbox.west - range)

                # ==============================================================
                # Rasterise processed source
                # ==============================================================
                r_source = "{}_{}_rast".format(TEMPNAME, src_cat)
                grass.run_command(
                    'v.to.rast',
                    input=v_sources,
                    type='area',
                    cats=str(src_cat),
                    output=r_source,
                    use='val',
                    overwrite=True,
                    quiet=True
                )

                # Check if raster contains any values
                univar = grass.read_command(
                            'r.univar',
                            map=r_source
                        )
                if int(univar.split('\n')[5].split(':')[1])==0:
                    continue

                # ==============================================================
                # Calculate cummulative (parametrised) viewshed from source
                # ==============================================================
                r_exposure = "{}_{}_exposure".format(TEMPNAME,src_cat)
                grass.run_command(
                    'r.viewshed.exposure',
                     dsm = r_dsm,
                     output = r_exposure,
                     source = r_source,
                     observer_elevation = v_elevation,
                     range = range,
                     function = function,
                     b1_distance = b_1,
                     sample_density = source_sample_density,
                     refraction_coeff = refr_coeff,
                     memory = memory,
                     cores = cores,
                     flags = flagstring,
                     quiet=True,
                     overwrite=True
                 )

                # ==============================================================
                # Convert to 0/1 if wanted, exclude tree pixels and apply weight
                # ==============================================================
                r_exposure_w = "{}_{}_exposure_weighted".format(TEMPNAME,src_cat)

                if binary_output:
                    expression = '$outmap = if(isnull($map_s),if($map_e > 0,$map_w,0),null())'

                else:
                    expression = '$outmap = if(isnull($map_s),$map_e * $map_w,null())'

                grass.mapcalc(
                    expression,
                    outmap=r_exposure_w,
                    map_s=r_source,
                    map_e=r_exposure,
                    map_w=r_weights,
                    overwrite=True
                )

                break

                #
                #
                # # ==========================================================================
                # # Summarise raster values and write to attribute table
                # # ==========================================================================
                # univar = grass.read_command(
                #             'r.univar',
                #             map=r_exposure_weighted
                #         )
                #
                # sum = float(univar.split('\n')[14].split(':')[1])
                #
                # writer.writerow({'src_cat':src_cat,
                #                  'value':sum})

    # close vector access
    vt_sources.close()


    # ## Restore original computational region
    # # gsl sets region for gsl tasks
    # grass.del_temp_region()
    #
    # # pygrass sets region for pygrass tasks
    # reg.read()
    # reg.set_current()
    # reg.set_raster_region()


    # Remove temporary files and reset mask if needed
    #cleanup()


if __name__ == '__main__':
    options, flags = grass.parser()
    #atexit.register(cleanup)
    sys.exit(main())
