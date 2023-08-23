"""Determine critical temperature by thresholding.

This script discards repeated temperature measurements due to Oxford software's
infrequent temperature updates from the Lakeshore resistance bridge.
"""

import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from shabanipy.labber import ShaBlabberFile
from shabanipy.utils import load_config, stamp

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("config_path", help="path to .ini config file")
parser.add_argument("config_section", help="section of the .ini config file to use")
args = parser.parse_args()
_, config = load_config(args.config_path, args.config_section)

with ShaBlabberFile(config["DATAPATH"]) as f:
    temp, volt = f.get_data(config["CH_TEMP"], config["CH_VOLT"])
    volt = volt.real

if temp.ndim > 1:
    # assume temperature is swept along last axis (i.e. Labber outer-most loop)
    # assume all other dimensions are trivial counters/timers
    temp = temp.flatten(order="F")
    volt = volt.flatten(order="F")

# remove stale data points due to Oxford control software pulling temperature data
# infrequently from Lakeshore resistance bridge
temp_fresh = [temp[0]]
indexs = [0]
for i, t in enumerate(temp):
    if t == temp_fresh[-1]:
        continue
    temp_fresh.append(t)
    indexs.append(i)
temp_fresh = np.array(temp_fresh)
volt_fresh = volt[indexs]

plt.style.use("fullscreen13")
fig, ax = plt.subplots()
ax.plot(temp, ".", label="stale")
ax.plot(indexs, temp_fresh, ".", label="fresh")
ax.set_ylabel(config["CH_TEMP"])
ax.set_title(args.config_section)
ax.legend()

sort_idx = np.argsort(temp_fresh)
temp_fresh = np.take_along_axis(temp_fresh, sort_idx, axis=0)
volt_fresh = np.take_along_axis(volt_fresh, sort_idx, axis=0)

fig, ax = plt.subplots()
ax.plot(temp, volt, ".", label="stale")
ax.plot(temp_fresh, volt_fresh, label="fresh")
ax.set_xlabel(config["CH_TEMP"])
ax.set_ylabel(config["CH_VOLT"])
ax.set_title(args.config_section)
ax.legend()

fig, ax = plt.subplots()
ax.plot(temp_fresh, volt_fresh / config.getfloat("IBIAS", 1), marker=".")
ax.set_xlabel("MXC temperature (K)")
ax.set_ylabel("resistance (Ω)" if config.get("IBIAS") else "voltage (V)")
ax.set_title(args.config_section)
stamp(ax, config["DATAPATH"])
fig.savefig(Path(config["DATAPATH"]).stem + "_" + args.config_section + "_Tc.png")

plt.show()
