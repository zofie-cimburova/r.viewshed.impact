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
import atexit
import sys
import subprocess
import numpy as np

from multiprocessing import Pool

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
R_DSM = None
RANGE = None
V_SRC = None
V_ELEVATION = None
FUNCTION = None
B_1 = None
REFR_COEFF = None
SOURCE_SAMPLE_DENSITY = None
SEED = None
MEMORY = None
CORES = None
FLGSTRING = None
R_WEIGHTS = None
BINARY_OUTPUT = None

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

def iteration(iterator):
    """Iterate over exposure source polygons, rasterise it, compute
    (paramterised) viewshed, exclude tree pixels, (convert to 0/1),
    (apply weight), summarise the value
    :param iterator: VectorTopo.viter object created from exposure source
    :type iterator:  VectorTopo.viter
    :return: String of cat,impact value
    :rtype: string
    """
    counter = 0
    string = ""

    for src in iterator:
        ## Display progress info message
        # TODO how to find length of viter?
        #grass.percent(counter, no_sources, 1)
        #counter += 1

        ## Only process features which have attribute table
        #TODO what are the features without attributes?
        if src.attrs is None:
            grass.verbose("Problem")
        else:
            src_cat = src.attrs['cat']
            grass.verbose('Processing source cat: {}'.format(src_cat))

            #if src_cat!= 95717:
            #    continue

            # ==============================================================
            # Set computational region to range around processed source
            # ==============================================================
            # TODO how to account for current settings of computational region?
            src_bbox = src.bbox()

            grass.run_command(
                'g.region',
                align=R_DSM,
                n=src_bbox.north + RANGE,
                s=src_bbox.south - RANGE,
                e=src_bbox.east + RANGE,
                w=src_bbox.west - RANGE
            )

            # ==============================================================
            # Rasterise processed source
            # ==============================================================
            r_source = "{}_{}_rast".format(TEMPNAME, src_cat)
            grass.run_command(
                'v.to.rast',
                input=V_SRC,
                type='area',
                cats=str(src_cat),
                output=r_source,
                use='val',
                overwrite=True,
                quiet=True
            )

            # Check if raster contains any values
            univar1 = grass.read_command(
                        'r.univar',
                        map=r_source
                    )
            if int(univar1.split('\n')[5].split(':')[1])==0:
                continue

            # ==============================================================
            # Calculate cummulative (parametrised) viewshed from source
            # ==============================================================
            r_exposure = "{}_{}_exposure".format(TEMPNAME,src_cat)
            grass.run_command(
                'r.viewshed.exposure',
                 dsm = R_DSM,
                 output = r_exposure,
                 source = r_source,
                 observer_elevation = V_ELEVATION,
                 range = RANGE,
                 function = FUNCTION,
                 b1_distance = B_1,
                 sample_density = SOURCE_SAMPLE_DENSITY,
                 refraction_coeff = REFR_COEFF,
                 seed = SEED,
                 memory = MEMORY,
                 cores = CORES,
                 flags = FLAGSTRING,
                 quiet=True,
                 overwrite=True
             )
             #TODO how to catch an exception when the tree is too small and
             #no sampling points are created?
             #(r.viewshed.exposure throws an error)

            # ==============================================================
            # Exclude tree pixels, (convert to 0/1), (apply weight)
            # ==============================================================
            r_exposure_w = "{}_{}_exposure_weighted".format(TEMPNAME,src_cat)

            if R_WEIGHTS:
                if BINARY_OUTPUT:
                    expression = '$out = if(isnull($s),if($e > 0,$w,0),null())'

                else:
                    expression = '$out = if(isnull($s),$e * $w,null())'
            else:
                if BINARY_OUTPUT:
                    expression = '$out = if(isnull($s),if($e > 0,1,0),null())'

                else:
                    expression = '$out = if(isnull($s),$e,null())'

            grass.mapcalc(
                expression,
                out=r_exposure_w,
                s=r_source,
                e=r_exposure,
                w=R_WEIGHTS,
                quiet=True,
                overwrite=True
            )

            # ==============================================================
            # Summarise raster values and write to string
            # ==============================================================
            univar2 = grass.read_command(
                        'r.univar',
                        map=r_exposure_w
                    )

            sum = float(univar2.split('\n')[14].split(':')[1])

            string += "{},{}\n".format(src_cat,sum)

            break
    return string


def f(xx):
    string = ""
    for x in xx:
        string += "{}\n".format(x*x)

    print(string)
    return string

def main():

    # set numpy printing options
    np.set_printoptions(formatter={'float': lambda x: '{0:0.2f}'.format(x)})

    # ==========================================================================
    # Input data
    # ==========================================================================
    ## DSM
    global R_DSM
    R_DSM = options['dsm']
    gfile_dsm = grass.find_file(name=R_DSM, element='cell')
    if not gfile_dsm['file']:
        grass.fatal('Raster map <%s> not found' % R_DSM)

    ## Exposure source
    global V_SRC
    V_SRC = options['exposure_source'].split("@")[0]

    #TODO why can only vector map in current mapset be used?
    gfile_source = grass.find_file(name=V_SRC, element='vector')
    if not gfile_source['file']:
        grass.fatal('Vector map <%s> not found' % V_SRC)

    # build topology in case it got corrupted
    grass.run_command('v.build',
                      map=V_SRC,
                      quiet=True)

    # convert to pygrass VectorTopo object
    v_src_topo = VectorTopo(V_SRC)
    v_src_topo.open('r')
    no_sources = v_src_topo.num_primitive_of('area')

    ## Weights
    global R_WEIGHTS
    R_WEIGHTS = options['weight']

    if R_WEIGHTS:
        gfile_weights = grass.find_file(name=R_WEIGHTS, element='cell')
        if not gfile_weights['file']:
            grass.fatal('Raster map <%s> not found' % R_WEIGHTS)

    ## Column to store visual impact values
    a_impact = options['column']

    #TODO ISSUE 6: how to check better if attribute already exists and what to do if it exists?
    # if a_impact in v_src_topo[1].attrs.keys():
    #     grass.fatal('Attribute <%s> already exists' % a_impact)
    # else:
    #     grass.run_command(
    #         'v.db.addcolumn',
    #         map=V_SRC,
    #         columns='{} double precision'.format(a_impact))


    ## Viewshed settings
    global FLAGSTRING
    FLAGSTRING = ''
    if flags['r']:
        FLAGSTRING += 'r'
    if flags['c']:
        FLAGSTRING += 'c'

    global V_ELEVATION
    V_ELEVATION = float(options['observer_elevation'])

    global RANGE
    RANGE = float(options['range_max'])

    global FUNCTION
    FUNCTION = options['function']

    global B_1
    B_1 = float(options['b1_distance'])

    global REFR_COEFF
    REFR_COEFF = float(options['refraction_coeff'])

    # test values
    if V_ELEVATION < 0.0:
        grass.fatal('Observer elevation must be larger than or equal to 0.0.')
    if RANGE <= 0.0 and RANGE != -1:
        grass.fatal('Maximum visibility radius must be larger than 0.0.')
    if FUNCTION == 'Fuzzy viewshed' and RANGE == -1:
        grass.fatal('Maximum visibility radius cannot be infinity for fuzzy viewshed approch.')
    if FUNCTION == 'Fuzzy viewshed' and B_1 > RANGE:
        grass.fatal(
            'Maximum visibility radius must be larger than radius around the viewpoint where clarity is perfect.'
        )

    # option for binary output instead of cummulative
    global BINARY_OUTPUT
    BINARY_OUTPUT = False
    if FUNCTION == "None":
        FUNCTION = "Binary"
        BINARY_OUTPUT = True

    ## Sampling settings
    global SOURCE_SAMPLE_DENSITY
    SOURCE_SAMPLE_DENSITY = float(options['sample_density'])

    global SEED
    SEED = options['seed']

    # if seed is not set, set it to process number
    if not SEED:
        SEED = os.getpid()

    ## Optional
    global MEMORY
    MEMORY = int(options['memory'])

    global CORES
    CORES = int(options['cores'])

    ## Keep or delete intermediate map
    global EXCLUDE
    if flags['w']:
        if R_WEIGHTS:
            EXCLUDE = "*_exposure_weighted"
        else:
            EXCLUDE = "*_exposure"
    # else:
        # #TODO how to use r.external.out with MEM option?
        # grass.run_command(
        #     "r.external.out",
        #     format="MEM"
        # )

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
    # Iteration over sources and computation of their visual impact
    # ==========================================================================
    src_iterator = v_src_topo.viter('areas')
    print(src_iterator)

    string = iteration(src_iterator)
    print(string)

    # close vector access
    v_src_topo.close()

    #pool = Pool(5)
    #string = pool.map(f, [[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    #pool.close()
    #pool.join() 
    #print(string)

    # ==============================================================
    # Write computed values to attribute table
    # ==============================================================
    #TODO - How to do?
    #grass.message(string)

    # ## Restore original computational region
    # # gsl sets region for gsl tasks
    # grass.del_temp_region()
    #
    # # pygrass sets region for pygrass tasks
    # reg.read()
    # reg.set_current()
    # reg.set_raster_region()

    # Restore storing in GRASS raster format
    # if !flags['w']:
    #     grass.run_command(
    #         flags="r"
    #     )

    # Remove temporary files and reset mask if needed
    cleanup()


if __name__ == '__main__':
    options, flags = grass.parser()
    atexit.register(cleanup)
    sys.exit(main())
