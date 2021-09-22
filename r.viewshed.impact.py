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
#% key: exposure
#% label: Name of input map of exposure source locations
#% required: yes
#% guidependency: range_layer, range_column
#%end

#%option G_OPT_V_FIELD
#% key: range_layer
#% guidependency: range_column
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
#% key: range_column
#% required: no
#% label: Name of attribute column containing exposure range
#% guisection: Viewshed settings
#%end

#%option
#% key: range
#% type: double
#% required: no
#% key_desc: value
#% label: Maximum exposure range
#% description: 0.0- , -1 for infinity
#% options: 0.0-
#% guisection: Viewshed settings
#%end

#%rules
#% required: range_column,range
#%end

#%rules
#% exclusive: range_column,range
#%end

#%option
#% key: function
#% type: string
#% required: no
#% key_desc: name
#% label: Viewshed parametrisation function
#% description: None, Binary, Distance_decay, Fuzzy_viewshed, Visual_magnitude, Solid_angle
#% options: None, Binary, Distance_decay, Fuzzy_viewshed, Visual_magnitude, Solid_angle
#% answer: Distance_decay
#% guisection: Viewshed settings
#%end

#%option
#% key: b1
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
#% key: refraction_coefficient
#% type: double
#% required: no
#% key_desc: value
#% label: Refraction coefficient
#% options: 0.0-1.0
#% description: 0.0-1.0
#% answer: 0.14286
#% guisection: Refraction
#%end

#%flag
#% key: k
#% label: Keep intermediate visual impact maps
#%end

#%flag
#% key: o
#% label: Allow intermediate visual impact maps to overwrite existing files
#%end

#%flag
#% key: a
#% label: Allow overwriting column storing visual impact values
#%end

#%option
#% key: prefix
#% required: no
#% label: Prefix for intermediate visual impact maps
#% answer: visual_impact_
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
import grass.script as grass

from multiprocessing import Pool
from grass.pygrass.gis.region import Region
from grass.pygrass.vector import VectorTopo
from grass.pygrass.raster import RasterRow
from grass.pygrass.gis import Mapset
from grass.script.raster import raster_info


# global variables
TEMPNAME = grass.tempname(12)
V_SRC = None
FUNCTION = None
SEED = None
BINARY_OUTPUT = None
REG = None
PREFIX = "visual_impact_"


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
    :param src: List of features
    :type src:  List
    :return: Sql command for upade of attribute table with visual impact value
    :rtype: String
    """

    # Category, range
    if options["range"] == "":
        cat = src[0]
        range = src[1]
    else:
        cat = src
        range = float(options["range"])

    # if cat not in [48814, 48815, 49466, 50499, 50612, 50813, 70865, 51428, 51428, 51462, 51477, 51567, 30452]:
    #     return None

    if range is None:
        sum = 0

        sql_command = (
            "UPDATE {table} SET {result_column} = {result} WHERE cat = {cat}".format(
                table=V_SRC, result_column=options["column"], result=sum, cat=cat
            )
        )
        return sql_command

    # Bounding box
    bbox_string = (
        grass.read_command(
            "v.db.select",
            map=V_SRC,
            where="cat={}".format(cat),
            flags="r",
        )
        .strip()
        .split("\n")
    )

    bbox = [
        float((bbox_string[0][2:])),
        float((bbox_string[1][2:])),
        float((bbox_string[3][2:])),
        float((bbox_string[2][2:])),
    ]

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
        align=options["dsm"],
    )

    # ==============================================================
    # Rasterise processed source
    # ==============================================================
    r_source = "{}_{}_rast".format(TEMPNAME, cat)
    grass.run_command(
        "v.to.rast",
        input=V_SRC,
        type="point,line,area,centroid",
        cats=str(cat),
        output=r_source,
        use="val",
        overwrite=True,
        quiet=True,
        env=env,
        stderr=subprocess.DEVNULL,
    )

    # Check if raster contains any values
    if raster_info(r_source)["max"] is None:
        sum = 0

        sql_command = (
            "UPDATE {table} SET {result_column} = {result} WHERE cat = {cat}".format(
                table=V_SRC, result_column=options["column"], result=sum, cat=cat
            )
        )
        return sql_command

    # ==============================================================
    # Distribute random sampling points (raster)
    # ==============================================================
    r_sample = "{}_{}_sample_rast".format(TEMPNAME, cat)
    grass.run_command(
        "r.random",
        input=r_source,
        raster=r_sample,
        npoints="{}%".format(options["sample_density"]),
        flags="b",
        overwrite=True,
        seed=SEED,
        quiet=True,
        env=env,
    )

    # Check if raster contains any values
    if raster_info(r_sample)["max"] is None:
        sum = 0

        sql_command = (
            "UPDATE {table} SET {result_column} = {result} WHERE cat = {cat}".format(
                table=V_SRC, result_column=options["column"], result=sum, cat=cat
            )
        )
        return sql_command

    # ==============================================================
    # Vectorize random sampling points
    # ==============================================================
    v_sample = "{}_{}_sample_vect".format(TEMPNAME, cat)
    grass.run_command(
        "r.to.vect",
        input=r_sample,
        output=v_sample,
        type="point",
        flags="bt",
        overwrite=True,
        quiet=True,
        env=env,
        stderr=subprocess.DEVNULL,
    )

    grass.run_command(
        "g.remove",
        flags="f",
        type="raster",
        name=r_sample,
        quiet=True,
        stderr=subprocess.PIPE,
    )

    # ==============================================================
    # Update processing environment with region information
    # around processed source
    # ==============================================================
    env["GRASS_REGION"] = grass.region_env(
        n=str(min(bbox[0] + range, REG.north)),
        s=str(max(bbox[1] - range, REG.south)),
        e=str(min(bbox[2] + range, REG.east)),
        w=str(max(bbox[3] - range, REG.west)),
        align=options["dsm"],
    )

    # ==============================================================
    # Calculate cummulative (parametrised) viewshed from source
    # ==============================================================
    flagstring = ""
    if flags["r"]:
        flagstring += "r"
    if flags["c"]:
        flagstring += "c"

    r_exposure = "{}_{}_exposure".format(TEMPNAME, cat)
    grass.run_command(
        "r.viewshed.exposure",
        dsm=options["dsm"],
        output=r_exposure,
        sampling_points=v_sample,
        observer_elevation=float(options["observer_elevation"]),
        range=range,
        function=FUNCTION,
        b1_distance=float(options["b1"]),
        refraction_coeff=float(options["refraction_coefficient"]),
        seed=SEED,
        memory=int(options["memory"]),
        cores=int(options["cores_e"]),
        flags=flagstring,
        overwrite=True,
        quiet=True,
        env=env,
    )

    grass.run_command(
        "g.remove",
        flags="f",
        type="vector",
        name=v_sample,
        quiet=True,
        stderr=subprocess.PIPE,
    )

    # ==============================================================
    # Exclude tree pixels, (convert to 0/1), (apply weight)
    # ==============================================================
    r_impact = "{}_{}_visual_impact".format(TEMPNAME, cat)

    if options["weight"] != "":
        if BINARY_OUTPUT:
            expression = (
                "$out = if(isnull($s),if(isnull($e),null(),if($e!=0,$w,0*$w)),null())"
            )
        else:
            expression = "$out = if(isnull($s),$e * $w,null())"
    else:
        if BINARY_OUTPUT:
            expression = (
                "$out = if(isnull($s),if(isnull($e),null(),if($e!=0,1,0)),null())"
            )
        else:
            expression = "$out = if(isnull($s),$e,null())"

    grass.mapcalc(
        expression,
        out=r_impact,
        s=r_source,
        e=r_exposure,
        w=options["weight"],
        quiet=True,
        overwrite=True,
        env=env,
    )

    grass.run_command(
        "g.remove",
        flags="f",
        type="raster",
        name=r_exposure,
        quiet=True,
        stderr=subprocess.PIPE,
    )

    # ==============================================================
    # Summarise impact value and write to string
    # ==============================================================
    univar = grass.read_command("r.univar", map=r_impact, env=env, quiet=True)
    sum = univar.split("\n")[14].split(":")[1]

    if sum in [" nan", " -nan"]:
        sum = 0.0
    elif sum in [" inf", " -inf", " Inf", " -Inf"]:
        return None
    else:
        sum = float(sum)

    sql_command = (
        "UPDATE {table} SET {result_column} = {result} WHERE cat = {cat}".format(
            table=V_SRC, result_column=options["column"], result=sum, cat=cat
        )
    )

    grass.verbose(sql_command)

    # ==============================================================
    # Rename visual impact map if it is to be kept
    # ==============================================================
    if flags["k"]:
        new_name = "{}{}".format(PREFIX, cat)

        if flags["o"]:
            grass.run_command(
                "g.rename",
                raster="{},{}".format(r_impact, new_name),
                overwrite=True,
                quiet=True,
                env=env,
            )
        else:
            gfile_vi = grass.find_file(name=new_name, element="cell")
            if gfile_vi["file"]:
                grass.warning("Raster map <%s> already exists. Skipping." % new_name)
            else:
                grass.run_command(
                    "g.rename",
                    raster="{},{}".format(r_impact, new_name),
                    overwrite=False,
                    quiet=True,
                    env=env,
                )
        grass.run_command(
            "g.remove",
            flags="f",
            type="raster",
            name=r_source,
            quiet=True,
            stderr=subprocess.PIPE,
        )
    else:
        grass.run_command(
            "g.remove",
            flags="f",
            type="raster",
            name="{},{}".format(r_source, r_impact),
            quiet=True,
            stderr=subprocess.PIPE,
        )

    return sql_command


def main():

    # ==========================================================================
    # Input data
    # ==========================================================================
    # DSM
    # check that the DSM exists
    gfile_dsm = grass.find_file(name=options["dsm"], element="cell")
    if not gfile_dsm["file"]:
        grass.fatal("Raster map <%s> not found" % options["dsm"])

    # EXPOSURE SOURCE VECTOR MAP
    global V_SRC
    V_SRC = options["exposure"].split("@")[0]
    mapset = options["exposure"].split("@")[1]

    # check that the vector map is in current mapset
    current_mapset = grass.read_command("g.mapset", flags="p").strip()
    if mapset != current_mapset:
        grass.fatal(
            "Vector map <{}> must be stored in current mapset <{}>".format(
                V_SRC, current_mapset
            )
        )

    # check that the vector map exists
    gfile_source = grass.find_file(name=V_SRC, element="vector")
    if not gfile_source["file"]:
        grass.fatal("Vector map <{}@{}> not found".format(V_SRC, mapset))

    # build topology of the vector map in case it got corrupted
    grass.run_command("v.build", map=V_SRC, quiet=True)

    # check that the vector map contains only point, line and area features
    info = grass.read_command("v.info", map=V_SRC, flags="t").strip().split("\n")
    # n_areas = int(info[5].split("=")[1])
    # n_boundaries = int(info[3].split("=")[1])
    # n_islands = int(info[6].split("=")[1])
    # n_map3d = int(info[8].split("=")[1])

    # if n_areas != n_boundaries:
    #     grass.fatal("r.viewshed.impact cannot process boundaries")
    # if n_areas != n_islands:
    #     grass.fatal("r.viewshed.impact cannot process islands")
    # if n_map3d > 0:
    #     grass.fatal("r.viewshed.impact cannot process map3d")

    # convert the vector map to pygrass VectorTopo object
    v_src_topo = VectorTopo(V_SRC)
    v_src_topo.open("r")

    # check that the weights map exists
    if options["weight"] != "":
        gfile_weights = grass.find_file(name=options["weight"], element="cell")
        if not gfile_weights["file"]:
            grass.fatal("Raster map <%s> not found" % options["weight"])

    # COLUMN TO STORE OUTPUT VISUAL IMPACT VALUE

    # check whether the column name contains allowed characters
    # check whether the column already exists in attribute table
    columns = grass.read_command("db.columns", table=V_SRC).strip().split("\n")

    if not grass.legal_name(options["column"]):
        grass.fatal("Invalid character in option 'column'.")

    elif options["column"] in columns:
        if flags["a"]:
            grass.warning(
                "Column <%s> already exists and will be overwritten" % options["column"]
            )
        else:
            grass.fatal("Column <%s> already exists" % options["column"])
    else:
        grass.run_command(
            "v.db.addcolumn",
            map=V_SRC,
            columns="{} double precision".format(options["column"]),
            quiet=True,
        )

    # OBSERVER ELEVATION
    if float(options["observer_elevation"]) < 0.0:
        grass.fatal("Observer elevation must be larger than or equal to 0.0.")

    # VIEWSHED PARAMETRISATION FUNCTION
    global FUNCTION
    FUNCTION = options["function"]

    # BINARY OUTPUT
    global BINARY_OUTPUT
    BINARY_OUTPUT = False

    if FUNCTION == "None":
        FUNCTION = "Binary"
        BINARY_OUTPUT = True

    # EXPOSURE RANGE - VALUE
    if options["range"] != "":
        if float(options["range"]) <= 0.0 and float(options["range"]) != -1:
            grass.fatal("Exposure range must be larger than 0.0.")

        if float(options["range"]) == -1 and FUNCTION == "Fuzzy_viewshed":
            grass.fatal(
                "Exposure range cannot be infinity for fuzzy viewshed function."
            )

        if (
            float(options["range"]) < float(options["b1"])
            and FUNCTION == "Fuzzy_viewshed"
        ):
            grass.fatal("Exposure range must be larger than b1.")

    # EXPOSURE RANGE - COLUMN
    if options["range_column"] != "":
        info = grass.read_command("v.info", flags="c", map=V_SRC, quiet=True).strip()
        info_dict = dict(reversed(i.split("|")) for i in info.split("\n"))

        # check if column exists
        if options["range_column"] not in info_dict:
            grass.fatal("Range column <%s> does not exist" % options["range_column"])

        # check if column is numeric
        if info_dict[options["range_column"]] not in ("INTEGER", "DOUBLE PRECISION"):
            grass.fatal("Range column <%s> must be numeric" % options["range_column"])

        # check that column values are nonnegative
        min = float(
            grass.read_command(
                "v.db.univar",
                flags="g",
                map=V_SRC,
                column=options["range_column"],
                quiet=True,
            )
            .strip()
            .split("\n")[1]
            .split("=")[1]
        )

        if min < 0:
            grass.fatal(
                "Range column <{}> must be nonnegative (min = {})".format(
                    options["range_column"], min
                )
            )

        if min < float(options["b1"]) and FUNCTION == "Fuzzy_viewshed":
            grass.fatal("Exposure range must be larger than b1.")

    # SEED
    global SEED
    SEED = options["seed"]

    if not SEED:
        SEED = os.getpid()

    # CORES
    cores_i = int(options["cores_i"])

    # NAME OF TEMPORARY MAPS
    global PREFIX

    if options["prefix"] == "":
        PREFIX = "visual_impact_"
    elif not grass.legal_name(options["prefix"]):
        grass.fatal("Invalid character in option 'prefix'.")
    else:
        PREFIX = options["prefix"]

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
    if abs(REG.nsres - REG.ewres) > 1e-6:
        grass.fatal(
            "Variable north-south and east-west 2D grid resolution is not supported"
        )

    # ==========================================================================
    # Iteration over features and computation of their visual impact
    # ==========================================================================
    # ensure that we only iterate over sources within computational region
    # use options["range_column"] if provided
    if options["range_column"] != "":
        features = {
            (ft.cat, ft.attrs[options["range_column"]])
            for ft in v_src_topo.find_by_bbox.geos(bbox=bbox)
            if ft.attrs is not None
        }
    else:
        features = {
            (ft.cat)
            for ft in v_src_topo.find_by_bbox.geos(bbox=bbox)
            if ft.attrs is not None
        }

    grass.verbose("Number of processed features: {}".format(len(features)))

    run_iteration = True
    if run_iteration:
        pool = Pool(cores_i)
        sql_list = pool.map(iteration, features)
        pool.close()
        pool.join()

    # close vector access
    v_src_topo.close()

    # ==============================================================
    # Write computed values to attribute table
    # ==============================================================
    grass.verbose("Writing output to attribute table...")
    write_result = True

    if write_result:
        for sql_command in sql_list:
            if sql_command:
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
