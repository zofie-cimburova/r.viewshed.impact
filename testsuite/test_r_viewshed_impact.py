#!/usr/bin/env python3

"""
MODULE:    Test of r.viewshed.impact
AUTHOR(S): Zofie Cimburova <stefan dot blumentrath at nina dot no>
           Stefan Blumentrath
PURPOSE:   Test of r.viewshed.exposure
COPYRIGHT: (C) 2020 by Zofie Cimburova, Stefan Blumentrath and the GRASS GIS
Development Team
This program is free software under the GNU General Public
License (>=v2). Read the file COPYING that comes with GRASS
for details.
"""

# python3 /home/NINA.NO/zofie.cimburova/git/r.viewshed.impact/testsuite/test_r_viewshed_impact.py

from grass.gunittest.gmodules import SimpleModule
import grass.script as gs

from grass.gunittest.case import TestCase
from grass.gunittest.main import test


class TestFunctions(TestCase):
    """The main (and only) test case for the r.viewshed.impact module"""

    tempname = gs.tempname(12)

    # maps used as inputs
    source_points = "schools_wake@PERMANENT"
    source_lines = "roadsmajor@PERMANENT"
    source_areas = "lakes@PERMANENT"
    weight = "aspect@PERMANENT"
    dsm = "elev_ned_30m@PERMANENT"

    # Check availability of r.viewshed.exposure and install it if missing
    # Assumes r.viewshed.exposure is part of the official repository
    if not gs.find_program("r.viewshed.exposure"):
        gs.run_command("g.extension", extension="r.viewshed.exposure")

    # copy exposure maps locally
    source_points_local = "schools_wake_local"
    source_lines_local = "roadsmajor_local"
    source_areas_local = "lakes_local"

    gs.run_command("g.copy", vector="{},{}".format(source_points, source_points_local))
    gs.run_command("g.copy", vector="{},{}".format(source_lines, source_lines_local))
    gs.run_command("g.copy", vector="{},{}".format(source_areas, source_areas_local))

    r_viewshed = SimpleModule(
        "r.viewshed.impact",
        # exposure=,
        # column="column_test",
        dsm=dsm,
        # weight=,
        flags="cra",
        observer_elevation=1.5,
        # range_column=,
        # range=,
        # function=,
        b1=90,
        # sample_density=100,
        seed=50,
        # flags="ko",
        memory=5000,
        cores_i=10,
        cores_e=2,
        quiet=True,
    )

    test_results_stats = {
        "test_point_b": {
            "n": 36,
            "min": 5859.57,
            "max": 44099,
            "sum": 849322,
            "mean": 23592.3,
        },
        "test_point_d": {
            "n": 36,
            "min": 0.0,
            "max": 16.7352,
            "sum": 381.836,
            "mean": 10.6066,
        },
        "test_line_f": {
            "n": 97,
            "min": 0.0,
            "max": 620.992,
            "sum": 8024.69,
            "mean": 82.7288,
        },
        "test_area_s": {
            "n": 793,
            "min": 0.0,
            "max": 0.753504,
            "sum": 2.54542,
            "mean": 0.00320987,
        },
        "test_area_v": {
            "n": 793,
            "min": 0.0,
            "max": 1.16588,
            "sum": 4.25116,
            "mean": 0.00536086,
        },
    }

    @classmethod
    def setUpClass(cls):
        """Save the current region to temporary file
        We cannot use temp_region as it is used by the module.
        """
        cls.runModule("g.region", flags="u", save="{}_region".format(cls.tempname))

    @classmethod
    def tearDownClass(cls):
        """Reset original region and remove the temporary region"""

        cls.runModule("g.region", region="{}_region".format(cls.tempname))
        cls.runModule(
            "g.remove", flags="f", type="region", name="{}_region".format(cls.tempname)
        )

        cls.runModule(
            "g.remove", flags="f", type="vector", name=cls.source_points_local
        )
        cls.runModule("g.remove", flags="f", type="vector", name=cls.source_lines_local)
        cls.runModule("g.remove", flags="f", type="vector", name=cls.source_areas_local)

    def test_points_b(self):
        """Test visibility of points, Binary, 300m"""

        # Use the input DSM to set computational region
        gs.run_command("g.region", raster=self.dsm, align=self.dsm)

        # Input parameters
        self.r_viewshed.inputs.exposure = self.source_points_local
        self.r_viewshed.inputs.weight = self.weight
        self.r_viewshed.inputs.function = "Binary"
        self.r_viewshed.inputs.range = 300
        self.r_viewshed.inputs.sample_density = 100
        self.r_viewshed.inputs.column = "column_test_b"

        # Print the command
        print(self.r_viewshed.get_bash())

        # Check that the module runs
        self.assertModule(self.r_viewshed)

        # Check if univariate vector statistics match the expected result
        self.assertVectorFitsUnivar(
            map=self.r_viewshed.inputs.exposure,
            column=self.r_viewshed.inputs.column,
            reference=self.test_results_stats["test_point_b"],
            precision=1e-4,
        )

    def test_points_d(self):
        """Test visibility of points, Distance decay, variable range"""

        # Use the input DSM to set computational region
        gs.run_command("g.region", raster=self.dsm, align=self.dsm)

        # Input parameters
        self.r_viewshed.inputs.exposure = self.source_points_local
        self.r_viewshed.inputs.function = "Distance_decay"
        self.r_viewshed.inputs.range_column = "PROJ_CAP"
        self.r_viewshed.inputs.sample_density = 100
        self.r_viewshed.inputs.range = None
        self.r_viewshed.inputs.column = "column_test_d"
        self.r_viewshed.inputs.weight = None

        # Print the command
        print(self.r_viewshed.get_bash())

        # Check that the module runs
        self.assertModule(self.r_viewshed)

        # Check if univariate vector statistics match the expected result
        self.assertVectorFitsUnivar(
            map=self.r_viewshed.inputs.exposure,
            column=self.r_viewshed.inputs.column,
            reference=self.test_results_stats["test_point_d"],
            precision=1e-4,
        )

    def test_lines_f(self):
        """Test visibility of lines, Fuzzy viewshed, 150m"""

        # Use the input DSM to set computational region
        gs.run_command("g.region", raster=self.dsm, align=self.dsm)

        # Input parameters
        self.r_viewshed.inputs.exposure = self.source_lines_local
        self.r_viewshed.inputs.function = "Fuzzy_viewshed"
        self.r_viewshed.inputs.range_column = None
        self.r_viewshed.inputs.range = 150
        self.r_viewshed.inputs.sample_density = 10
        self.r_viewshed.inputs.column = "column_test_f"
        self.r_viewshed.inputs.weight = None

        # Print the command
        print(self.r_viewshed.get_bash())

        # Check that the module runs
        self.assertModule(self.r_viewshed)

        # Check if univariate vector statistics match the expected result
        self.assertVectorFitsUnivar(
            map=self.r_viewshed.inputs.exposure,
            column=self.r_viewshed.inputs.column,
            reference=self.test_results_stats["test_line_f"],
            precision=1e-4,
        )

    def test_areas_s(self):
        """Test visibility of areas, Solid angle, 150m"""

        # Use the input DSM to set computational region
        gs.run_command("g.region", raster=self.dsm, align=self.dsm)

        # Input parameters
        self.r_viewshed.inputs.exposure = self.source_areas_local
        self.r_viewshed.inputs.function = "Solid_angle"
        self.r_viewshed.inputs.range_column = None
        self.r_viewshed.inputs.range = 150
        self.r_viewshed.inputs.sample_density = 1
        self.r_viewshed.inputs.column = "column_test_s"
        self.r_viewshed.inputs.weight = None

        # Print the command
        print(self.r_viewshed.get_bash())

        # Check that the module runs
        self.assertModule(self.r_viewshed)

        # Check if univariate vector statistics match the expected result
        self.assertVectorFitsUnivar(
            map=self.r_viewshed.inputs.exposure,
            column=self.r_viewshed.inputs.column,
            reference=self.test_results_stats["test_area_s"],
            precision=1e-4,
        )

    def test_areas_v(self):
        """Test visibility of areas, Visual magnitude, 150m"""

        # Use the input DSM to set computational region
        gs.run_command("g.region", raster=self.dsm, align=self.dsm)

        # Input parameters
        self.r_viewshed.inputs.exposure = self.source_areas_local
        self.r_viewshed.inputs.function = "Visual_magnitude"
        self.r_viewshed.inputs.range_column = None
        self.r_viewshed.inputs.range = 150
        self.r_viewshed.inputs.sample_density = 1
        self.r_viewshed.inputs.column = "column_test_v"
        self.r_viewshed.inputs.weight = None

        # Print the command
        print(self.r_viewshed.get_bash())

        # Check that the module runs
        self.assertModule(self.r_viewshed)

        # Check if univariate vector statistics match the expected result
        self.assertVectorFitsUnivar(
            map=self.r_viewshed.inputs.exposure,
            column=self.r_viewshed.inputs.column,
            reference=self.test_results_stats["test_area_v"],
            precision=1e-4,
        )


if __name__ == "__main__":
    test()
