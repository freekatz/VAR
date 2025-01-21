#!/usr/bin/env bash

set -x

version=$1

PWD=$(cd "$(dirname "$0")"/../;pwd)
cd $PWD
rm -rf output
mkdir -p output
cd output
cp -r ../Makefile ./
cp -r ../hack ./
cp -r ../utils ./
cp -r ../models ./
cp -r ../scripts ./
cp -r ../*.py ./

cd ../

fname=var_v${version}
mv ./output ./${fname}
zip -r ./${fname}.zip ./${fname}/*
mv ./${fname} ./output
mkdir -p release
mv ./${fname}.zip ./release
