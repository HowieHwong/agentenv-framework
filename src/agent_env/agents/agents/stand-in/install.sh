#!/bin/sh
# Installs the stand-in onto this Linux host from the folder this script is in, and starts it on PORT, as USER in
# WORKDIR, the place of the agent it replaces:
#
#   install.sh NAME PORT WORKDIR USER
#
# It runs in a venv of the host's own Python, which apt installs when the host has none.
set -eu
name=$1 port=$2 workdir=$3 user=$4
here=$(cd "$(dirname "$0")" && pwd)
root=/opt/agentenv-stand-in-$name

if ! python3 -c "import ensurepip, venv" 2>/dev/null; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends python3 python3-venv
fi
rm -rf "$root"
mkdir -p "$root"
cp "$here/a2a_server.py" "$here/requirements.txt" "$root/"
python3 -m venv "$root/venv"
"$root/venv/bin/pip" install -q -r "$root/requirements.txt"
id -u "$user" >/dev/null 2>&1 || useradd -m "$user"
mkdir -p "$workdir"
chown -R "$user": "$workdir"
nohup runuser -u "$user" -- env -C "$workdir" A2A_PORT="$port" "$root/venv/bin/python" "$root/a2a_server.py" \
    >"/tmp/agentenv-stand-in-$name.log" 2>&1 </dev/null &
