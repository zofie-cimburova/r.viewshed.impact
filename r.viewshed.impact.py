#!/usr/bin/env python3

"""
MODULE:       r.viewshed.impact

AUTHOR(S):    Zofie Cimburova, Stefan Blumentrath

PURPOSE:      Computes visual impact of defined exposure source using weighted parametrised viewshed analysis

COPYRIGHT:    (C) 2021 by Zofie Cimburova, Stefan Blumentrath, and the GRASS Development Team

REFERENCES:   TODO reference papers used and the paper to be published

This program is free software under the GNU General Public
License (>=v2). Read the file COPYING that comes with GRASS
for details.
"""
# python3 /home/NINA.NO/zofie.cimburova/PhD/Paper4/SRC/r.viewshed.impact.py

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

#%rules
#% required: range_col,range_max
#%end

#%rules
#% exclusive: range_col,range_max
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
#% label: Consider the effect of atmospheric refraction in viewshed modelling
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
#% key: cores_e
#% type: integer
#% required: no
#% key_desc: value
#% label: Number of cores to parallelise r.viewshed.exposure
#% description: 1-
#% options: 1-
#% answer: 1
#%end

#%option
#% key: cores_i
#% type: integer
#% required: no
#% key_desc: value
#% label: Number of cores to parallelise r.viewshed.impact
#% description: 1-
#% options: 1-
#% answer: 1
#%end

import os
import atexit
import sys
import subprocess
import numpy as np
import grass.script as grass

from multiprocessing import Pool
from grass.pygrass.gis.region import Region
from grass.pygrass.vector import VectorTopo
from grass.pygrass.raster import RasterRow
from grass.pygrass.gis import Mapset
from grass.script.raster import raster_info


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
CORES_E = None
FLGSTRING = None
R_WEIGHTS = None
BINARY_OUTPUT = None
REG = None
OVERWRITE = None
COLUMN = None


def cleanup():
    """Remove temporary raster and vector maps"""
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
        except Exception:
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
        except Exception:
            pass


def iteration(src):
    """Iterate over exposure source polygons, rasterise it, compute
    (paramterised) viewshed, exclude tree pixels, (convert to 0/1),
    (apply weight), summarise the value
    :param src: List of areas
    :type src:  List
    :return: Sql command for upade of attribute table with visual impact value
    :rtype: String
    """

    cat = src[0]
    bbox = src[1]

    # Adjust range
    if len(src) == 3:
        range = src[2]
    else:
        range = RANGE

    # Display progress info message
    grass.verbose("Processing source cat: {}".format(cat))

    # Tempname
    suffix = grass.tempname(3)[4:]

    # ==============================================================
    # Create processing environment with region information
    # around processed source
    # ==============================================================
    # ensure that local region doesn't exceed global region
    env = os.environ.copy()
    env["GRASS_REGION"] = grass.region_env(
        n=str(min(bbox[0], REG.north)),
        s=str(max(bbox[1], REG.south)),
        e=str(min(bbox[2], REG.east)),
        w=str(max(bbox[3], REG.west)),
        align=R_DSM,
    )

    # ==============================================================
    # Rasterise processed source
    # ==============================================================
    r_source = "{}_{}_{}_rast".format(TEMPNAME, cat, suffix)
    grass.run_command(
        "v.to.rast",
        input=V_SRC,
        type="area,centroid",
        cats=str(cat),
        output=r_source,
        use="val",
        overwrite=True,
        quiet=True,
        env=env,
    )

    # Check if raster contains any values
    if raster_info(r_source)["max"] is None:
        string = "{},{}\n".format(cat, 0)
        return string

    # ==============================================================
    # Distribute random sampling points (raster)
    # ==============================================================
    r_sample = "{}_{}_{}_sample_rast".format(TEMPNAME, cat, suffix)
    grass.run_command(
        "r.random",
        input=r_source,
        raster=r_sample,
        npoints="{}%".format(SOURCE_SAMPLE_DENSITY),
        flags="b",
        overwrite=True,
        quiet=True,
        env=env,
    )

    # Check if raster contains any values
    if raster_info(r_sample)["max"] is None:
        string = "{},{}\n".format(cat, 0)
        return string

    # ==============================================================
    # Distribute random sampling points (vector)
    # ==============================================================
    v_sample = "{}_{}_{}_sample_vect".format(TEMPNAME, cat, suffix)
    p = grass.start_command(
        "r.to.vect",
        input=r_sample,
        output=v_sample,
        type="point",
        flags="bt",
        overwrite=True,
        quiet=True,
        env=env,
        stderr=subprocess.PIPE,
    )

    err_msg = str(p.communicate()[1])
    if "ERROR" in err_msg:
        grass.fatal(err_msg)
        # TODO - how to handle this situation? What to do with pool?

    # ==============================================================
    # Update processing environment with region information
    # around processed source
    # ==============================================================
    env["GRASS_REGION"] = grass.region_env(
        n=str(min(bbox[0] + range, REG.north)),
        s=str(max(bbox[1] - range, REG.south)),
        e=str(min(bbox[2] + range, REG.east)),
        w=str(max(bbox[3] - range, REG.west)),
        align=R_DSM,
    )

    # ==============================================================
    # Calculate cummulative (parametrised) viewshed from source
    # ==============================================================
    r_exposure = "{}_{}_{}_exposure".format(TEMPNAME, cat, suffix)
    grass.run_command(
        "r.viewshed.exposure",
        dsm=R_DSM,
        output=r_exposure,
        sampling_points=v_sample,
        observer_elevation=V_ELEVATION,
        range=range,
        function=FUNCTION,
        b1_distance=B_1,
        sample_density=SOURCE_SAMPLE_DENSITY,
        refraction_coeff=REFR_COEFF,
        seed=SEED,
        memory=MEMORY,
        cores=CORES_E,
        flags=FLAGSTRING,
        overwrite=True,
        quiet=True,
        env=env,
    )

    # ==============================================================
    # Exclude tree pixels, (convert to 0/1), (apply weight)
    # ==============================================================
    r_impact = "{}_{}_{}_visual_impact".format(TEMPNAME, cat, suffix)

    if R_WEIGHTS:
        if BINARY_OUTPUT:
            expression = "$out = if(isnull($s),if($e > 0,$w,0),null())"
        else:
            expression = "$out = if(isnull($s),$e * $w,null())"
    else:
        if BINARY_OUTPUT:
            expression = "$out = if(isnull($s),if($e > 0,1,0),null())"
        else:
            expression = "$out = if(isnull($s),$e,null())"

    grass.mapcalc(
        expression,
        out=r_impact,
        s=r_source,
        e=r_exposure,
        w=R_WEIGHTS,
        quiet=True,
        overwrite=True,
        env=env,
    )

    # ==============================================================
    # Summarise impact value and write to string
    # ==============================================================
    univar = grass.read_command(
        "r.univar",
        map=r_impact,
        env=env,
    )

    sum = float(univar.split("\n")[14].split(":")[1])
    sql_command = (
        "UPDATE {table} SET {result_column} = {result} WHERE cat = {cat}".format(
            table=V_SRC, result_column=COLUMN, result=sum, cat=cat
        )
    )

    # ==============================================================
    # Rename visual impact map if it is to be kept
    # ==============================================================
    # TODO how to distinguish between suffix in new name?

    if EXCLUDE == 1:
        new_name = "visual_impact_{}".format(cat)
        grass.run_command(
            "g.rename",
            raster="{},{}".format(r_impact, new_name),
            overwrite=OVERWRITE,
            quiet=True,
            env=env,
        )

    return sql_command


def main():

    # set numpy printing options
    np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})

    # ==========================================================================
    # Input data
    # ==========================================================================
    # DSM
    global R_DSM
    R_DSM = options["dsm"]
    gfile_dsm = grass.find_file(name=R_DSM, element="cell")
    if not gfile_dsm["file"]:
        grass.fatal("Raster map <%s> not found" % R_DSM)

    # Exposure source
    global V_SRC
    V_SRC = options["exposure_source"].split("@")[0]

    # TODO why can only vector map in current mapset be used?
    gfile_source = grass.find_file(name=V_SRC, element="vector")
    if not gfile_source["file"]:
        grass.fatal("Vector map <%s> not found" % V_SRC)

    # build topology in case it got corrupted
    grass.run_command("v.build", map=V_SRC, quiet=True)

    # convert to pygrass VectorTopo object
    v_src_topo = VectorTopo(V_SRC)
    v_src_topo.open("r")

    # Weights
    global R_WEIGHTS
    R_WEIGHTS = options["weight"]

    if R_WEIGHTS:
        gfile_weights = grass.find_file(name=R_WEIGHTS, element="cell")
        if not gfile_weights["file"]:
            grass.fatal("Raster map <%s> not found" % R_WEIGHTS)

    # Column to store visual impact values
    global COLUMN
    COLUMN = options["column"]

    # TODO how to check better if attribute already exists and what to do if it exists?
    # if a_impact in v_src_topo[1].attrs.keys():
    #     grass.fatal("Attribute <%s> already exists" % a_impact)
    # else:
    #     grass.run_command(
    #         "v.db.addcolumn",
    #         map=V_SRC,
    #         columns="{} double precision".format(a_impact))

    # Viewshed settings
    global FLAGSTRING
    FLAGSTRING = ""
    if flags["r"]:
        FLAGSTRING += "r"
    if flags["c"]:
        FLAGSTRING += "c"

    global V_ELEVATION
    V_ELEVATION = float(options["observer_elevation"])

    global RANGE
    RANGE = float(options["range_max"])
    range_col = options["range_col"]

    global FUNCTION
    FUNCTION = options["function"]

    global B_1
    B_1 = float(options["b1_distance"])

    global REFR_COEFF
    REFR_COEFF = float(options["refraction_coeff"])

    global OVERWRITE
    OVERWRITE = grass.overwrite()

    # test values
    if V_ELEVATION < 0.0:
        grass.fatal("Observer elevation must be larger than or equal to 0.0.")
    if RANGE <= 0.0 and RANGE != -1:
        grass.fatal("Maximum visibility radius must be larger than 0.0.")
    if FUNCTION == "Fuzzy viewshed" and RANGE == -1:
        grass.fatal(
            "Maximum visibility radius cannot be infinity for fuzzy viewshed approch."
        )
    if FUNCTION == "Fuzzy viewshed" and B_1 > RANGE:
        grass.fatal(
            "Maximum visibility radius must be larger than radius around the viewpoint where clarity is perfect."
        )

    # option for binary output instead of cummulative
    global BINARY_OUTPUT
    BINARY_OUTPUT = False
    if FUNCTION == "None":
        FUNCTION = "Binary"
        BINARY_OUTPUT = True

    # Sampling settings
    global SOURCE_SAMPLE_DENSITY
    SOURCE_SAMPLE_DENSITY = float(options["sample_density"])

    global SEED
    SEED = options["seed"]

    # if seed is not set, set it to process number
    if not SEED:
        SEED = os.getpid()

    # Optional
    global MEMORY
    MEMORY = int(options["memory"])

    global CORES_E
    CORES_E = int(options["cores_e"])

    cores_i = int(options["cores_i"])

    # Keep or delete intermediate map
    global EXCLUDE
    if flags["w"]:
        EXCLUDE = 1
    else:
        EXCLUDE = 0

    # ==========================================================================
    # Region and mask settings
    # ==========================================================================
    # check that location is not in lat/long
    if grass.locn_is_latlong():
        grass.fatal("The analysis is not available for lat/long coordinates.")

    if RasterRow("MASK", Mapset().name).exist():
        grass.warning(_("Current MASK is temporarily renamed."))
        unset_mask()

    # get comp. region parameters
    global REG
    REG = Region()
    bbox = REG.get_bbox()

    # check that nsres equals ewres
    if REG.nsres != REG.ewres:
        grass.fatal(
            "Variable north-south and east-west 2D grid resolution is not supported"
        )

    # ==========================================================================
    # Iteration over sources and computation of their visual impact
    # ==========================================================================
    # ensure that we only iterate over sources within computational region
    # use range_col if provided
    if range_col != "":
        src_areas = [
            (area.centroid().cat, area.bbox().nsewtb(tb=False), area.attrs[range_col])
            for area in v_src_topo.find_by_bbox.areas(bbox=bbox)
            if area.attrs is not None
        ]
    else:
        src_areas = [
            (area.centroid().cat, area.bbox().nsewtb(tb=False))
            for area in v_src_topo.find_by_bbox.areas(bbox=bbox)
            if area.attrs is not None
        ]

    # Sequential
    # string=""
    # for src in src_areas:
    #    string += iteration(src)
    #    break

    # Parallel
    pool = Pool(cores_i)
    sql_list = pool.map(iteration, src_areas)
    pool.close()
    pool.join()

    # close vector access
    v_src_topo.close()

    # ==============================================================
    # Write computed values to attribute table
    # ==============================================================
    for sql_command in sql_list:
        grass.run_command(
            "db.execute",
            sql=sql_command,
        )

    # Remove temporary files and reset mask if needed
    cleanup()


if __name__ == "__main__":
    options, flags = grass.parser()
    atexit.register(cleanup)
    sys.exit(main())
