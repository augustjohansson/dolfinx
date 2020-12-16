# -*- coding: utf-8 -*-
# Copyright (C) 2016-2016 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os
import pathlib
import subprocess
import sys

import pytest

# Get directory of this file
path = pathlib.Path(__file__).resolve().parent

# Build list of demo programs
demos = []
demo_files = list(path.glob('**/*.py'))
for f in demo_files:
    demos.append((f.parent, f.name))


@pytest.mark.serial
@pytest.mark.parametrize("path,name", demos)
def test_demos(path, name):
    if name in ["demo_elasticity.py", "path4-demo_gmsh.py", "demo_stokes-taylor-hood.py"]:
        pytest.skip()

    ret = subprocess.run([sys.executable, name],
                         cwd=str(path),
                         env={**os.environ, 'MPLBACKEND': 'agg'},
                         check=True)
    assert ret.returncode == 0


@pytest.mark.mpi
@pytest.mark.parametrize("path,name", demos)
def test_demos_mpi(num_proc, mpiexec, path, name):
    if name in ["demo_elasticity.py", "path4-demo_gmsh.py", "demo_stokes-taylor-hood.py"]:
        pytest.skip()

    cmd = [mpiexec, "-np", str(num_proc), sys.executable, name]
    print(cmd)
    ret = subprocess.run(cmd,
                         cwd=str(path),
                         env={**os.environ, 'MPLBACKEND': 'agg'},
                         check=True)
    assert ret.returncode == 0
