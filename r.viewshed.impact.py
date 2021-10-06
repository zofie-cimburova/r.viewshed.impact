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
import itertools

from multiprocessing import Pool
from grass.pygrass.gis.region import Region
from grass.pygrass.vector import VectorTopo
from grass.pygrass.raster import RasterRow
from grass.pygrass.gis import Mapset
from grass.script.raster import raster_info


# global variables
TEMPNAME = grass.tempname(12)


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


def iteration(global_vars, src):
    """Iterate over exposure source polygons, rasterise it, compute
    (paramterised) viewshed, exclude tree pixels, (convert to 0/1),
    (apply weight), summarise the value
    :param src: List of features
    :type src:  List
    :return: Sql command for upade of attribute table with visual impact value
    :rtype: String
    """

    # Get variables out of global_vars dictionary
    exp_range = global_vars["range"]
    v_src = global_vars["v_src"]
    reg = global_vars["reg"]
    dsm = global_vars["dsm"]
    sample_density = global_vars["sample_density"]
    column = global_vars["column"]
    observer_elevation = global_vars["observer_elevation"]
    b_1 = global_vars["b_1"]
    refr_coeff = global_vars["refr_coeff"]
    memory = global_vars["memory"]
    cores = global_vars["cores"]
    weight = global_vars["weight"]
    function = global_vars["function"]
    seed = global_vars["seed"]
    binary_output = global_vars["binary_output"]
    prefix = global_vars["prefix"]
    flagstring = global_vars["flagstring"]
    keep_tmp = global_vars["keep_tmp"]
    overwrite_tmp = global_vars["overwrite_tmp"]

    # Category, range
    if exp_range == "":
        cat = src[0]
        range = src[1]
    else:
        cat = src
        range = float(exp_range)

    if range is None:
        sum = 0

        sql_command = (
            "UPDATE {table} SET {result_column} = {result} WHERE cat = {cat}".format(
                table=v_src, result_column=column, result=sum, cat=cat
            )
        )
        return sql_command

    # Bounding box
    bbox_string = grass.parse_command(
        "v.db.select",
        map=v_src,
        where="cat={}".format(cat),
        flags="r",
    )

    bbox = [
        float((bbox_string["n"])),
        float((bbox_string["s"])),
        float((bbox_string["e"])),
        float((bbox_string["w"])),
    ]

    # ==============================================================
    # Create processing environment with region information
    # around processed source
    # ==============================================================
    # ensure that local region doesn't exceed global region
    env = os.environ.copy()
    env["GRASS_REGION"] = grass.region_env(
        n=str(min(bbox[0], reg.north)),
        s=str(max(bbox[1], reg.south)),
        e=str(min(bbox[2], reg.east)),
        w=str(max(bbox[3], reg.west)),
        align=dsm,
    )

    # ==============================================================
    # Rasterise processed source
    # ==============================================================
    r_source = "{}_{}_rast".format(TEMPNAME, cat)
    grass.run_command(
        "v.to.rast",
        input=v_src,
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
                table=v_src, result_column=column, result=sum, cat=cat
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
        npoints="{}%".format(sample_density),
        flags="b",
        overwrite=True,
        seed=seed,
        quiet=True,
        env=env,
    )

    # Check if raster contains any values
    if raster_info(r_sample)["max"] is None:
        sum = 0

        sql_command = (
            "UPDATE {table} SET {result_column} = {result} WHERE cat = {cat}".format(
                table=v_src, result_column=column, result=sum, cat=cat
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
        n=str(min(bbox[0] + range, reg.north)),
        s=str(max(bbox[1] - range, reg.south)),
        e=str(min(bbox[2] + range, reg.east)),
        w=str(max(bbox[3] - range, reg.west)),
        align=dsm,
    )

    # ==============================================================
    # Calculate cummulative (parametrised) viewshed from source
    # ==============================================================
    r_exposure = "{}_{}_exposure".format(TEMPNAME, cat)
    grass.run_command(
        "r.viewshed.exposure",
        dsm=dsm,
        output=r_exposure,
        sampling_points=v_sample,
        observer_elevation=float(observer_elevation),
        range=range,
        function=function,
        b1_distance=float(b_1),
        refraction_coeff=float(refr_coeff),
        seed=seed,
        memory=int(memory),
        cores=int(cores),
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

    if weight != "":
        if binary_output:
            expression = (
                "$out = if(isnull($s),if(isnull($e),null(),if($e!=0,$w,0*$w)),null())"
            )
        else:
            expression = "$out = if(isnull($s),$e * $w,null())"
    else:
        if binary_output:
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
        w=weight,
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
    sum = grass.parse_command(
        "r.univar",
        map=r_impact,
        flags="g",
        env=env,
        quiet=True,
    )["sum"]

    if sum in [" nan", " -nan"]:
        sum = 0.0
    elif sum in [" inf", " -inf", " Inf", " -Inf"]:
        return None
    else:
        sum = float(sum)

    sql_command = (
        "UPDATE {table} SET {result_column} = {result} WHERE cat = {cat}".format(
            table=v_src, result_column=column, result=sum, cat=cat
        )
    )

    grass.verbose(sql_command)

    # ==============================================================
    # Rename visual impact map if it is to be kept
    # ==============================================================
    if keep_tmp:
        new_name = "{}{}".format(prefix, cat)

        if overwrite_tmp:
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
    dsm = options["dsm"]
    # check that the DSM exists
    gfile_dsm = grass.find_file(name=dsm, element="cell")
    if not gfile_dsm["file"]:
        grass.fatal("Raster map <%s> not found" % dsm)

    # EXPOSURE SOURCE VECTOR MAP
    v_src = options["exposure"].split("@")[0]
    mapset = options["exposure"].split("@")[1]

    # check that the vector map is in current mapset
    current_mapset = grass.read_command("g.mapset", flags="p").strip()
    if mapset != current_mapset:
        grass.fatal(
            "Vector map <{}> must be stored in current mapset <{}>".format(
                v_src, current_mapset
            )
        )

    # check that the vector map exists
    gfile_source = grass.find_file(name=v_src, element="vector")
    if not gfile_source["file"]:
        grass.fatal("Vector map <{}@{}> not found".format(v_src, mapset))

    # build topology of the vector map in case it got corrupted
    grass.run_command("v.build", map=v_src, quiet=True)

    # check that the vector map contains only point, line and area features
    # info = grass.read_command("v.info", map=v_src, flags="t").strip().split("\n")
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
    v_src_topo = VectorTopo(v_src)
    v_src_topo.open("r")

    # check that the weights map exists
    weight = options["weight"]
    if weight != "":
        gfile_weights = grass.find_file(name=weight, element="cell")
        if not gfile_weights["file"]:
            grass.fatal("Raster map <%s> not found" % weight)

    # COLUMN TO STORE OUTPUT VISUAL IMPACT VALUE

    # check whether the column name contains allowed characters
    # check whether the column already exists in attribute table
    columns = grass.read_command("db.columns", table=v_src).strip().split("\n")
    column = options["column"]

    if not grass.legal_name(column):
        grass.fatal("Invalid character in option 'column'.")

    elif column in columns:
        if flags["a"]:
            grass.warning("Column <%s> already exists and will be overwritten" % column)
        else:
            grass.fatal("Column <%s> already exists" % column)
    else:
        grass.run_command(
            "v.db.addcolumn",
            map=v_src,
            columns="{} double precision".format(column),
            quiet=True,
        )

    # OBSERVER ELEVATION
    observer_elevation = options["observer_elevation"]
    if float(observer_elevation) < 0.0:
        grass.fatal("Observer elevation must be larger than or equal to 0.0.")

    # VIEWSHED PARAMETRISATION FUNCTION
    function = options["function"]

    # BINARY OUTPUT
    binary_output = False

    if function == "None":
        function = "Binary"
        binary_output = True

    # EXPOSURE RANGE - VALUE
    exp_range = options["range"]
    b_1 = options["b1"]
    if exp_range != "":
        if float(exp_range) <= 0.0 and float(exp_range) != -1:
            grass.fatal("Exposure range must be larger than 0.0.")

        if float(exp_range) == -1 and function == "Fuzzy_viewshed":
            grass.fatal(
                "Exposure range cannot be infinity for fuzzy viewshed function."
            )

        if float(exp_range) < float(b_1) and function == "Fuzzy_viewshed":
            grass.fatal("Exposure range must be larger than b1.")

    # EXPOSURE RANGE - COLUMN
    if options["range_column"] != "":
        info = grass.read_command("v.info", flags="c", map=v_src, quiet=True).strip()
        info_dict = dict(reversed(i.split("|")) for i in info.split("\n"))

        # check if column exists
        if options["range_column"] not in info_dict:
            grass.fatal("Range column <%s> does not exist" % options["range_column"])

        # check if column is numeric
        if info_dict[options["range_column"]] not in ("INTEGER", "DOUBLE PRECISION"):
            grass.fatal("Range column <%s> must be numeric" % options["range_column"])

        # check that column values are nonnegative
        min = float(
            grass.parse_command(
                "v.db.univar",
                flags="g",
                map=v_src,
                column=options["range_column"],
                quiet=True,
            )["min"]
        )

        if min < 0:
            grass.fatal(
                "Range column <{}> must be nonnegative (min = {})".format(
                    options["range_column"], min
                )
            )

        if min < float(b_1) and function == "Fuzzy_viewshed":
            grass.fatal("Exposure range must be larger than b1.")

    # SEED
    seed = options["seed"]

    if not seed:
        seed = os.getpid()

    # CORES
    cores_i = int(options["cores_i"])

    # NAME OF TEMPORARY MAPS
    prefix = "visual_impact_"

    if options["prefix"] == "":
        prefix = "visual_impact_"
    elif not grass.legal_name(options["prefix"]):
        grass.fatal("Invalid character in option 'prefix'.")
    else:
        prefix = options["prefix"]

    # FLAGSTRING
    flagstring = ""
    if flags["r"]:
        flagstring += "r"
    if flags["c"]:
        flagstring += "c"

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
    reg = Region()
    bbox = reg.get_bbox()

    # check that nsres equals ewres
    if abs(reg.nsres - reg.ewres) > 1e-6:
        grass.fatal(
            "Variable north-south and east-west 2D grid resolution is not supported"
        )

    # ==========================================================================
    # Iteration over features and computation of their visual impact
    # ==========================================================================

    # Collect variables that will be used in do_it_all() into a dictionary
    global_vars = {
        "range": exp_range,
        "v_src": v_src,
        "reg": reg,
        "dsm": dsm,
        "sample_density": options["sample_density"],
        "column": column,
        "observer_elevation": observer_elevation,
        "b_1": b_1,
        "refr_coeff": options["refraction_coefficient"],
        "memory": options["memory"],
        "cores": options["cores_e"],
        "weight": weight,
        "function": function,
        "seed": seed,
        "binary_output": binary_output,
        "prefix": prefix,
        "flagstring": flagstring,
        "keep_tmp": flags["k"],
        "overwrite_tmp": flags["o"],
    }

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

    combo = list(zip(itertools.repeat(global_vars), features))

    run_iteration = True
    if run_iteration:
        pool = Pool(cores_i)
        sql_list = pool.starmap(iteration, combo)
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
